;; Export docs/orgmode/*.org to website/ (ox-html).
;; Usage: emacs --batch -l docs/export.el
(require 'ox-html)

;; Org draws anchor ids from the PRNG, so every export rewrites them and
;; the tracked HTML churns. Derive them from the heading path instead.
;; A heading with a CUSTOM_ID property keeps that id.
(defvar minimage--export-datum nil
  "Element or object whose reference `org-export-get-reference' is building.")

(defun minimage--datum-key (datum)
  "Return a string that identifies DATUM independently of the PRNG."
  (if (eq (org-element-type datum) 'headline)
      (let (titles)
        (while datum
          (when (eq (org-element-type datum) 'headline)
            (push (org-element-property :raw-value datum) titles))
          (setq datum (org-element-property :parent datum)))
        (mapconcat #'identity titles "/"))
    (format "%s@%s" (org-element-type datum)
            (org-element-property :begin datum))))

(define-advice org-export-get-reference
    (:around (fn datum info) minimage-stable-id)
  (let ((minimage--export-datum datum))
    (funcall fn datum info)))

(define-advice org-export-new-reference
    (:around (fn references) minimage-stable-id)
  (if (null minimage--export-datum)
      (funcall fn references)
    (let ((new (string-to-number
                (substring (secure-hash 'sha1 (minimage--datum-key
                                               minimage--export-datum))
                           0 7)
                16)))
      (while (rassq new references)
        (setq new (% (1+ new) #x10000000)))
      new)))

;; Org 9.7 stores the blank line after a headline or section on that
;; element's last child. Org 9.6 stores it on the headline or section.
;; ox-html keeps a parent's :post-blank as a blank line between sections
;; and drops a child's :post-blank when it normalizes the section body.
;; Emacs 30.2 (Org 9.7.11) therefore drops the blank lines Emacs 29.3
;; (Org 9.6.15) keeps. The tracked HTML follows the blank lines in the
;; Org source. No export option chooses which element owns the blank, so
;; lift it onto the parent before HTML export. Both versions then match.
(defun minimage-normalize-section-gap (tree _backend _info)
  "Pin the blank line between sections on the parent element."
  (minimage--lift-post-blank tree)
  tree)

(defun minimage--lift-post-blank (elem)
  "Move a trailing :post-blank from the last child of ELEM onto ELEM."
  (dolist (child (org-element-contents elem))
    (minimage--lift-post-blank child))
  (when (memq (org-element-type elem) '(headline section))
    (let ((last (car (last (org-element-contents elem)))))
      (when (and last
                 (> (or (org-element-property :post-blank last) 0) 0)
                 (= (or (org-element-property :post-blank elem) 0) 0))
        (org-element-put-property elem :post-blank
                                  (org-element-property :post-blank last))
        (org-element-put-property last :post-blank 0)))))

(setq org-export-with-toc nil
      org-html-prefer-user-labels t
      org-export-with-section-numbers nil
      org-export-with-author nil
      org-export-with-timestamps nil
      org-export-time-stamp-file nil
      org-export-with-sub-superscripts nil
      org-html-head-include-default-style nil
      org-html-head-include-scripts nil
      org-html-validation-link nil
      org-html-creator-string nil
      org-html-preamble nil
      org-html-postamble nil
      org-html-doctype "html5"
      org-html-html5-fancy t
      org-html-htmlize-output-type nil
      org-html-container-element "section"
      org-html-divs
      '((preamble "div" "preamble")
        (content "div" "content")
        (postamble "div" "postamble"))
      org-export-filter-parse-tree-functions
      (cons #'minimage-normalize-section-gap
            org-export-filter-parse-tree-functions)
      org-html-text-markup-alist
      '((bold . "<strong>%s</strong>")
        (code . "<code>%s</code>")
        (italic . "<em>%s</em>")
        (strike-through . "<del>%s</del>")
        (underline . "<span class=\"underline\">%s</span>")
        (verbatim . "<code>%s</code>")))

(let* ((this (file-name-directory (or load-file-name buffer-file-name)))
       (org-dir (expand-file-name "orgmode" this))
       (out-dir (expand-file-name "../website" this))
       (pages '("index.org" "getting-started.org" "tutorial.org"
                "howto.org" "explanation.org" "simd.org" "reference.org")))
  (unless (file-directory-p out-dir)
    (make-directory out-dir t))
  (dolist (page pages)
    (find-file (expand-file-name page org-dir))
    (org-html-export-as-html)
    (write-file (expand-file-name
                 (concat (file-name-sans-extension page) ".html")
                 out-dir))
    (kill-buffer)))
