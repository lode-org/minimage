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
