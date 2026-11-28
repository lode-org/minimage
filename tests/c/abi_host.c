/* A host that compiles against the minimage headers alone and takes the
 * library at run time: no minimage symbol is linked. The path of the
 * shared library is argv[1], or MINIMAGE_LIBRARY. The ABI the header
 * names is checked against the library before any other call. Every
 * operation then comes from the table, whose null members are operations
 * the library lacks, and each result is compared with the entry points a
 * linked host calls directly, on orthorhombic, sheared, slab, and wire
 * cells, including cells built with negative edges.
 * Exit 0 on success; a failure prints its line and exits 1. */
#include <dlfcn.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "minimage.h"

/* The frozen layouts of layout revision 1. A change here is a new
 * revision, which a library of the old one refuses. */
_Static_assert(MINIMAGE_ABI_LAYOUT_REVISION == 1, "these are revision 1");
_Static_assert(sizeof(minimage_abi) == 12, "minimage_abi is frozen");
_Static_assert(offsetof(minimage_abi, major) == 0, "minimage_abi.major");
_Static_assert(offsetof(minimage_abi, minor) == 4, "minimage_abi.minor");
_Static_assert(offsetof(minimage_abi, layout_revision) == 8,
               "minimage_abi.layout_revision");
/* Every other struct starts with its uint32_t size ... */
_Static_assert(offsetof(minimage_caps, size) == 0, "minimage_caps.size");
_Static_assert(offsetof(minimage_ops, size) == 0, "minimage_ops.size");
_Static_assert(offsetof(minimage_lattice_info, size) == 0,
               "minimage_lattice_info.size");
_Static_assert(sizeof(((minimage_caps *)0)->size) == 4, "size is 32 bits");
_Static_assert(sizeof(((minimage_ops *)0)->size) == 4, "size is 32 bits");
_Static_assert(sizeof(((minimage_lattice_info *)0)->size) == 4,
               "size is 32 bits");
/* ... and the members it already has keep their offsets as it grows. */
_Static_assert(offsetof(minimage_caps, lattice) == 4, "caps.lattice");
_Static_assert(offsetof(minimage_caps, open_axes) == 8, "caps.open_axes");
_Static_assert(offsetof(minimage_caps, wrap_diffs) == 12, "caps.wrap_diffs");
_Static_assert(offsetof(minimage_caps, wrap_pairs) == 16, "caps.wrap_pairs");
_Static_assert(offsetof(minimage_caps, nearest_diffs) == 20,
               "caps.nearest_diffs");
_Static_assert(offsetof(minimage_caps, nearest_pairs) == 24,
               "caps.nearest_pairs");
_Static_assert(offsetof(minimage_caps, wrap_points) == 28,
               "caps.wrap_points");
_Static_assert(offsetof(minimage_caps, in_place) == 32, "caps.in_place");
_Static_assert(offsetof(minimage_caps, wrap_open_sheared) == 36,
               "caps.wrap_open_sheared");
_Static_assert(sizeof(minimage_caps) >= 40, "caps grows, never shrinks");
_Static_assert(offsetof(minimage_lattice_info, orthorhombic) == 4,
               "info.orthorhombic");
_Static_assert(offsetof(minimage_lattice_info, periodic) == 8,
               "info.periodic");
_Static_assert(offsetof(minimage_lattice_info, reserved) == 20,
               "info.reserved");
_Static_assert(offsetof(minimage_lattice_info, rows) == 24, "info.rows");
_Static_assert(offsetof(minimage_lattice_info, origin) == 96, "info.origin");
_Static_assert(offsetof(minimage_lattice_info, widths) == 120,
               "info.widths");
_Static_assert(sizeof(minimage_lattice_info) >= 144,
               "info grows, never shrinks");
#define OPS_SLOT(member, index)                                       \
    _Static_assert(offsetof(minimage_ops, member) ==                  \
                       8 + (index) * sizeof(void (*)(void)),          \
                   "minimage_ops." #member)
_Static_assert(offsetof(minimage_ops, reserved) == 4, "ops.reserved");
OPS_SLOT(last_error, 0);
OPS_SLOT(version, 1);
OPS_SLOT(capabilities, 2);
OPS_SLOT(lattice_diagonal, 3);
OPS_SLOT(lattice_general, 4);
OPS_SLOT(lattice_free, 5);
OPS_SLOT(lattice_get_info, 6);
OPS_SLOT(lattice_wrap_diffs, 7);
OPS_SLOT(lattice_wrap_pairs, 8);
OPS_SLOT(lattice_nearest_diffs, 9);
OPS_SLOT(lattice_nearest_pairs, 10);
OPS_SLOT(lattice_wrap_points, 11);
_Static_assert(sizeof(minimage_ops) >= 8 + 12 * sizeof(void (*)(void)),
               "ops grows, never shrinks");

#define CHECK(cond)                                                          \
    do {                                                                     \
        if (!(cond)) {                                                       \
            fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, \
                    #cond);                                                  \
            exit(1);                                                         \
        }                                                                    \
    } while (0)

/* The three entry points of the negotiation, and the entry points a
 * linked host calls directly, all resolved by name. */
static struct {
    int32_t (*abi_stamp)(minimage_abi *);
    int32_t (*abi_compatible)(const minimage_abi *);
    int32_t (*ops_get)(minimage_ops *);
    int (*cell_from_vectors)(const double *, const double *, const double *,
                             const double *, mi_cell *);
    int (*wrap_many)(const mi_cell *, const double *, size_t, double *);
    int (*displacement)(const mi_cell *, const double *, const double *,
                        double *);
    int (*displacement_euclidean)(const mi_cell *, const double *,
                                  const double *, double *);
    int (*pbc_cell_from_ase)(const double *, const int *, const double *,
                             mi_pbc_cell *);
    int (*displacement_euclidean_pbc)(const mi_pbc_cell *, const double *,
                                      const double *, double *);
} direct;

static minimage_ops ops;

static void *symbol(void *lib, const char *name) {
    void *found = dlsym(lib, name);
    if (found == NULL) {
        fprintf(stderr, "missing symbol %s\n", name);
        exit(1);
    }
    return found;
}

/* A function pointer through an object pointer, as dlsym returns it. */
#define LOAD(member, name)                          \
    do {                                            \
        void *sym_ = symbol(lib, name);             \
        memcpy(&direct.member, &sym_, sizeof sym_); \
    } while (0)

static uint64_t rng_state = UINT64_C(0x9e3779b97f4a7c15);

/* A double in [-scale, scale). */
static double next(double scale) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return ((double)(rng_state >> 11) / 9007199254740992.0 * 2.0 - 1.0) *
           scale;
}

static int close_to(double a, double b) {
    const double scale = fabs(a) > fabs(b) ? fabs(a) : fabs(b);
    return fabs(a - b) <= 1e-9 * (1.0 + scale);
}

enum { N = 600 };

static double ps[N * 3], qs[N * 3], diffs[N * 3];

static void fill_points(double scale) {
    for (size_t k = 0; k < N * 3; ++k) {
        ps[k] = next(scale);
        qs[k] = next(scale);
        diffs[k] = qs[k] - ps[k];
    }
}

static void expect_ok(int32_t status) {
    if (status != MINIMAGE_STATUS_OK) {
        const char *text = ops.last_error();
        fprintf(stderr, "status %d: %s\n", (int)status,
                text != NULL ? text : "(no message)");
        exit(1);
    }
    CHECK(ops.last_error() == NULL);
}

/* The negotiation: the stamp, the compatibility rule, and the two
 * size-prefixed structs as an older and a newer host see them. */
static void negotiate(void) {
    minimage_abi lib_abi = {99, 99, 99};
    CHECK(direct.abi_stamp(&lib_abi) == MINIMAGE_STATUS_OK);
    printf("header ABI %u.%u layout %u, library ABI %u.%u layout %u\n",
           (unsigned)MINIMAGE_ABI_MAJOR, (unsigned)MINIMAGE_ABI_MINOR,
           (unsigned)MINIMAGE_ABI_LAYOUT_REVISION, (unsigned)lib_abi.major,
           (unsigned)lib_abi.minor, (unsigned)lib_abi.layout_revision);
    CHECK(lib_abi.major == MINIMAGE_ABI_MAJOR);
    CHECK(lib_abi.layout_revision == MINIMAGE_ABI_LAYOUT_REVISION);
    CHECK(direct.abi_stamp(NULL) == MINIMAGE_STATUS_NULL_POINTER);

    minimage_abi host = {MINIMAGE_ABI_MAJOR, MINIMAGE_ABI_MINOR,
                         MINIMAGE_ABI_LAYOUT_REVISION};
    CHECK(direct.abi_compatible(&host) == MINIMAGE_STATUS_OK);
    /* The minor is informational. */
    host.minor += 5;
    CHECK(direct.abi_compatible(&host) == MINIMAGE_STATUS_OK);
    host.major += 1;
    CHECK(direct.abi_compatible(&host) == MINIMAGE_STATUS_INCOMPATIBLE);
    host.major -= 1;
    host.layout_revision += 1;
    CHECK(direct.abi_compatible(&host) == MINIMAGE_STATUS_INCOMPATIBLE);
    CHECK(direct.abi_compatible(NULL) == MINIMAGE_STATUS_NULL_POINTER);

    /* The table as this header knows it. */
    memset(&ops, 0, sizeof ops);
    ops.size = (uint32_t)sizeof ops;
    CHECK(direct.ops_get(&ops) == MINIMAGE_STATUS_OK);
    CHECK(ops.size == sizeof ops);
    CHECK(ops.last_error != NULL && ops.version != NULL);
    CHECK(ops.capabilities != NULL && ops.lattice_diagonal != NULL);
    CHECK(ops.lattice_general != NULL && ops.lattice_free != NULL);
    CHECK(ops.lattice_get_info != NULL && ops.lattice_wrap_diffs != NULL);
    CHECK(ops.lattice_wrap_pairs != NULL && ops.lattice_nearest_diffs != NULL);
    CHECK(ops.lattice_nearest_pairs != NULL);
    CHECK(ops.lattice_wrap_points != NULL);
    printf("library version %s\n", ops.version());

    /* An older host that knows the first two operations: the bytes it
     * declared are written and nothing past them. */
    minimage_ops older;
    memset(&older, 0xff, sizeof older);
    older.size = (uint32_t)offsetof(minimage_ops, capabilities);
    CHECK(direct.ops_get(&older) == MINIMAGE_STATUS_OK);
    CHECK(older.size == offsetof(minimage_ops, capabilities));
    CHECK(older.last_error == ops.last_error && older.version == ops.version);
    unsigned char untouched[sizeof older.capabilities];
    memset(untouched, 0xff, sizeof untouched);
    CHECK(memcmp(&older.capabilities, untouched, sizeof untouched) == 0);

    /* A newer host whose table has operations this library lacks: they
     * read as null, which is how a host learns they are unsupported. */
    struct {
        minimage_ops known;
        void (*later[3])(void);
    } newer;
    memset(&newer, 0xff, sizeof newer);
    newer.known.size = (uint32_t)sizeof newer;
    CHECK(direct.ops_get(&newer.known) == MINIMAGE_STATUS_OK);
    CHECK(newer.known.size == sizeof ops);
    CHECK(newer.known.lattice_wrap_points == ops.lattice_wrap_points);
    for (size_t k = 0; k < 3; ++k) {
        CHECK(newer.later[k] == NULL);
    }

    minimage_ops tiny;
    tiny.size = 2;
    CHECK(direct.ops_get(&tiny) == MINIMAGE_STATUS_BAD_SIZE);
    CHECK(ops.last_error() != NULL);
    CHECK(direct.ops_get(NULL) == MINIMAGE_STATUS_NULL_POINTER);

    /* The capabilities, in the same three ways. */
    minimage_caps caps;
    memset(&caps, 0, sizeof caps);
    caps.size = (uint32_t)sizeof caps;
    expect_ok(ops.capabilities(&caps));
    CHECK(caps.size == sizeof caps);
    CHECK(caps.lattice == 1 && caps.open_axes == 1 && caps.in_place == 1);
    CHECK(caps.wrap_diffs == 1 && caps.wrap_pairs == 1);
    CHECK(caps.nearest_diffs == 1 && caps.nearest_pairs == 1);
    CHECK(caps.wrap_points == 1 && caps.wrap_open_sheared == 0);

    minimage_caps fewer;
    memset(&fewer, 0xff, sizeof fewer);
    fewer.size = (uint32_t)offsetof(minimage_caps, wrap_diffs);
    expect_ok(ops.capabilities(&fewer));
    CHECK(fewer.size == offsetof(minimage_caps, wrap_diffs));
    CHECK(fewer.lattice == 1 && fewer.open_axes == 1);
    CHECK(fewer.wrap_diffs == UINT32_MAX && fewer.in_place == UINT32_MAX);

    struct {
        minimage_caps known;
        uint32_t later[4];
    } more;
    memset(&more, 0xff, sizeof more);
    more.known.size = (uint32_t)sizeof more;
    expect_ok(ops.capabilities(&more.known));
    CHECK(more.known.size == sizeof caps);
    for (size_t k = 0; k < 4; ++k) {
        CHECK(more.later[k] == 0);
    }
    caps.size = 3;
    CHECK(ops.capabilities(&caps) == MINIMAGE_STATUS_BAD_SIZE);
    CHECK(ops.capabilities(NULL) == MINIMAGE_STATUS_NULL_POINTER);
}

static minimage_lattice *build(const double *rows, const uint32_t *periodic,
                               const double *origin,
                               minimage_lattice_info *info) {
    minimage_lattice *lattice = NULL;
    expect_ok(ops.lattice_general(rows, periodic, origin, &lattice));
    CHECK(lattice != NULL);
    memset(info, 0, sizeof *info);
    info->size = (uint32_t)sizeof *info;
    expect_ok(ops.lattice_get_info(lattice, info));
    CHECK(info->size == sizeof *info);
    return lattice;
}

/* A fully periodic lattice against the calls a linked host makes on the
 * mi_cell of the same vectors. */
static void check_full(const double *rows, const char *name) {
    static const double origin[3] = {0.3, -1.1, 2.0};
    minimage_lattice_info info;
    minimage_lattice *lattice = build(rows, NULL, origin, &info);
    CHECK(info.periodic[0] == 1 && info.periodic[1] == 1);
    CHECK(info.periodic[2] == 1);

    /* The direct cell, from the caller's rows: the library normalises it
     * the same way, so both report one set of vectors. */
    mi_cell cell;
    CHECK(direct.cell_from_vectors(rows, rows + 3, rows + 6, origin, &cell) ==
          0);
    const double reported[9] = {cell.ax, cell.ay, cell.az, cell.bx, cell.by,
                                cell.bz, cell.cx, cell.cy, cell.cz};
    CHECK(memcmp(reported, info.rows, sizeof reported) == 0);
    if (info.orthorhombic == 1) {
        CHECK(info.rows[0] > 0.0 && info.rows[4] > 0.0 && info.rows[8] > 0.0);
        CHECK(info.widths[0] == info.rows[0]);
        CHECK(info.widths[1] == info.rows[4]);
        CHECK(info.widths[2] == info.rows[8]);
    }

    fill_points(25.0);
    static double out[N * 3], want[N * 3], same[N * 3];

    /* The engine wrap is minimage_wrap_many, bit for bit. */
    expect_ok(ops.lattice_wrap_diffs(lattice, diffs, N, out));
    CHECK(direct.wrap_many(&cell, diffs, N, want) == 0);
    CHECK(memcmp(out, want, sizeof out) == 0);
    /* In place it is the same wrap. */
    memcpy(same, diffs, sizeof same);
    expect_ok(ops.lattice_wrap_diffs(lattice, same, N, same));
    for (size_t k = 0; k < N * 3; ++k) {
        CHECK(close_to(same[k], want[k]));
    }

    /* Pairs are minimage_displacement. */
    expect_ok(ops.lattice_wrap_pairs(lattice, ps, qs, N, out));
    for (size_t k = 0; k < N; ++k) {
        double dr[3];
        CHECK(direct.displacement(&cell, ps + 3 * k, qs + 3 * k, dr) == 0);
        for (size_t c = 0; c < 3; ++c) {
            CHECK(close_to(out[3 * k + c], dr[c]));
        }
    }

    /* The nearest image is minimage_displacement_euclidean, bit for bit. */
    expect_ok(ops.lattice_nearest_pairs(lattice, ps, qs, N, out));
    for (size_t k = 0; k < N; ++k) {
        CHECK(direct.displacement_euclidean(&cell, ps + 3 * k, qs + 3 * k,
                                            want + 3 * k) == 0);
    }
    CHECK(memcmp(out, want, sizeof out) == 0);
    expect_ok(ops.lattice_nearest_diffs(lattice, diffs, N, same));
    for (size_t k = 0; k < N * 3; ++k) {
        CHECK(close_to(same[k], want[k]));
    }

    /* A wrapped point is an image of its point: the engine wrap of the
     * move is zero. It is also a fixed point of the wrap. */
    expect_ok(ops.lattice_wrap_points(lattice, ps, N, out));
    for (size_t k = 0; k < N; ++k) {
        double dr[3];
        CHECK(direct.displacement(&cell, ps + 3 * k, out + 3 * k, dr) == 0);
        CHECK(fabs(dr[0]) < 1e-9 && fabs(dr[1]) < 1e-9 && fabs(dr[2]) < 1e-9);
    }
    expect_ok(ops.lattice_wrap_points(lattice, out, N, same));
    CHECK(memcmp(out, same, sizeof out) == 0);
    if (info.orthorhombic == 1) {
        for (size_t k = 0; k < N; ++k) {
            for (size_t c = 0; c < 3; ++c) {
                const double x = out[3 * k + c] - origin[c];
                CHECK(x >= 0.0 && x < info.rows[4 * c]);
            }
        }
    }

    ops.lattice_free(lattice);
    printf("%s: %d rows match the direct calls\n", name, (int)N);
}

/* A lattice with open axes against minimage_displacement_euclidean_pbc. */
static void check_mixed(const double *rows, const uint32_t *periodic,
                        const char *name) {
    minimage_lattice_info info;
    minimage_lattice *lattice = build(rows, periodic, NULL, &info);
    const int mask[3] = {(int)periodic[0], (int)periodic[1],
                         (int)periodic[2]};
    mi_pbc_cell cell;
    CHECK(direct.pbc_cell_from_ase(rows, mask, NULL, &cell) == 0);

    fill_points(25.0);
    static double out[N * 3], want[N * 3];
    expect_ok(ops.lattice_nearest_pairs(lattice, ps, qs, N, out));
    for (size_t k = 0; k < N; ++k) {
        CHECK(direct.displacement_euclidean_pbc(&cell, ps + 3 * k, qs + 3 * k,
                                                want + 3 * k) == 0);
    }
    for (size_t k = 0; k < N * 3; ++k) {
        CHECK(close_to(out[k], want[k]));
    }

    const int32_t status = ops.lattice_wrap_diffs(lattice, diffs, N, out);
    if (info.orthorhombic == 1) {
        /* Per axis: the open component as given, the periodic one the
         * nearest image. */
        expect_ok(status);
        for (size_t k = 0; k < N; ++k) {
            for (size_t c = 0; c < 3; ++c) {
                if (periodic[c] == 0) {
                    CHECK(out[3 * k + c] == diffs[3 * k + c]);
                } else {
                    CHECK(close_to(out[3 * k + c], want[3 * k + c]));
                }
            }
        }
        static double folded[N * 3];
        expect_ok(ops.lattice_wrap_points(lattice, ps, N, folded));
        for (size_t k = 0; k < N; ++k) {
            for (size_t c = 0; c < 3; ++c) {
                const double x = folded[3 * k + c];
                if (periodic[c] == 0) {
                    CHECK(x == ps[3 * k + c]);
                } else {
                    CHECK(x >= 0.0 && x < info.rows[4 * c]);
                }
            }
        }
    } else {
        /* No engine wrap for a sheared lattice with an open axis: the
         * call says so and writes nothing. */
        memset(out, 0, sizeof out);
        CHECK(ops.lattice_wrap_diffs(lattice, diffs, N, out) ==
              MINIMAGE_STATUS_UNSUPPORTED);
        CHECK(status == MINIMAGE_STATUS_UNSUPPORTED);
        CHECK(ops.last_error() != NULL);
        for (size_t k = 0; k < N * 3; ++k) {
            CHECK(out[k] == 0.0);
        }
    }
    ops.lattice_free(lattice);
    printf("%s: %d rows match the direct calls\n", name, (int)N);
}

static void check_failures(void) {
    const double good[3] = {10.0, 10.0, 10.0};
    const double zero[3] = {10.0, 0.0, 10.0};
    const double nan_edge[3] = {10.0, 10.0, NAN};
    const uint32_t slab[3] = {1, 0, 1};
    minimage_lattice *lattice = (minimage_lattice *)&direct;

    CHECK(ops.lattice_diagonal(zero, NULL, NULL, &lattice) ==
          MINIMAGE_STATUS_BAD_CELL);
    CHECK(lattice == NULL && ops.last_error() != NULL);
    CHECK(ops.lattice_diagonal(nan_edge, NULL, NULL, &lattice) ==
          MINIMAGE_STATUS_BAD_CELL);
    /* The zero edge is on the open axis of this slab. */
    expect_ok(ops.lattice_diagonal(zero, slab, NULL, &lattice));
    ops.lattice_free(lattice);
    CHECK(ops.lattice_diagonal(NULL, NULL, NULL, &lattice) ==
          MINIMAGE_STATUS_NULL_POINTER);
    CHECK(ops.lattice_diagonal(good, NULL, NULL, NULL) ==
          MINIMAGE_STATUS_NULL_POINTER);

    expect_ok(ops.lattice_diagonal(good, NULL, NULL, &lattice));
    double rows[15] = {0};
    CHECK(ops.lattice_wrap_diffs(NULL, rows, 4, rows) ==
          MINIMAGE_STATUS_NULL_POINTER);
    CHECK(ops.lattice_wrap_diffs(lattice, NULL, 4, rows) ==
          MINIMAGE_STATUS_NULL_POINTER);
    CHECK(ops.lattice_wrap_diffs(lattice, rows, SIZE_MAX / 8, rows) ==
          MINIMAGE_STATUS_BAD_COUNT);
    CHECK(ops.lattice_wrap_diffs(lattice, rows, 4, rows + 3) ==
          MINIMAGE_STATUS_BAD_OVERLAP);
    CHECK(ops.lattice_wrap_pairs(lattice, rows, rows, 4, rows + 3) ==
          MINIMAGE_STATUS_BAD_OVERLAP);
    /* No rows: the buffers are not read. */
    expect_ok(ops.lattice_wrap_diffs(lattice, NULL, 0, NULL));
    expect_ok(ops.lattice_nearest_pairs(lattice, NULL, NULL, 0, NULL));
    ops.lattice_free(lattice);
    ops.lattice_free(NULL);
    printf("failures: each cause has its own code\n");
}

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : getenv("MINIMAGE_LIBRARY");
    if (path == NULL) {
        fprintf(stderr, "usage: %s <path to libminimage>\n", argv[0]);
        return 2;
    }
    void *lib = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (lib == NULL) {
        fprintf(stderr, "dlopen %s: %s\n", path, dlerror());
        return 1;
    }
    LOAD(abi_stamp, "minimage_abi_stamp");
    LOAD(abi_compatible, "minimage_abi_compatible");
    LOAD(ops_get, "minimage_ops_get");
    LOAD(cell_from_vectors, "minimage_cell_from_vectors");
    LOAD(wrap_many, "minimage_wrap_many");
    LOAD(displacement, "minimage_displacement");
    LOAD(displacement_euclidean, "minimage_displacement_euclidean");
    LOAD(pbc_cell_from_ase, "minimage_pbc_cell_from_ase");
    LOAD(displacement_euclidean_pbc, "minimage_displacement_euclidean_pbc");

    negotiate();

    static const double cube[9] = {15, 0, 0, 0, 15, 0, 0, 0, 15};
    static const double reported[9] = {-15, 0, 0, 0, -15, 0, 0, 0, 15};
    static const double all_negative[9] = {-15, 0, 0, 0, -12, 0, 0, 0, -9};
    static const double sheared[9] = {10, 0, 0, 3, 9, 0, 1, 2, 8};
    static const double sheared_negative[9] = {-10, 0, 0, 3, -9, 0, 1, 2, -8};
    static const double left_handed[9] = {3, 9, 0, 10, 0, 0, 1, 2, 8};
    check_full(cube, "cube");
    check_full(reported, "cube with two negative edges");
    check_full(all_negative, "box with three negative edges");
    check_full(sheared, "sheared");
    check_full(sheared_negative, "sheared with a negative diagonal");
    check_full(left_handed, "left-handed");

    static const uint32_t slab[3] = {1, 1, 0};
    static const uint32_t wire[3] = {0, 0, 1};
    static const uint32_t open[3] = {0, 0, 0};
    check_mixed(cube, slab, "orthorhombic slab");
    check_mixed(reported, slab, "orthorhombic slab with negative edges");
    check_mixed(all_negative, wire, "orthorhombic wire with a negative edge");
    check_mixed(cube, open, "open");
    check_mixed(sheared, slab, "sheared slab");
    check_mixed(sheared_negative, wire, "sheared wire");

    check_failures();
    dlclose(lib);
    printf("abi host: ok\n");
    return 0;
}
