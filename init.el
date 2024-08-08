;; Needed for package: exec-path-from-shell
(when (memq window-system '(mac ns x))
  (exec-path-from-shell-initialize))

;; inihibit splash screen at startup
(setq inhibit-splash-screen t)

;; maximize screen on startup
(add-hook 'window-setup-hook 'toggle-frame-maximized t)

;; Eglot hooks
(add-hook 'python-mode-hook 'eglot-ensure)
(add-hook 'zig-mode-hook 'eglot-ensure)

;; Company-mode in all buffers
(add-hook 'after-init-hook 'global-company-mode)


;; disable menu on startup
(menu-bar-mode -1)

;;disable tools on startup
(tool-bar-mode -1)

;; disable scroll bar on startup
(scroll-bar-mode -1)

;;Display line number mode
(global-display-line-numbers-mode 1)
(setq display-line-numbers-type 'relative)

;;(ido-mode 1)

;; turn on electric pair mode (useful when typing opening and closing brackets, cursor moves automatically inside the brackets)
(electric-pair-mode 1)

;; Set column width to 80
(setq-default fill-column 80)


(custom-set-variables
 ;; custom-set-variables was added by Custom.
 ;; If you edit it by hand, you could mess it up, so be careful.
 ;; Your init file should contain only one such instance.
 ;; If there is more than one, they won't work right.
 '(custom-enabled-themes '(gruber-darker))
 '(custom-safe-themes
   '("e27c9668d7eddf75373fa6b07475ae2d6892185f07ebed037eedf783318761d7"
     "871b064b53235facde040f6bdfa28d03d9f4b966d8ce28fb1725313731a2bcc8"
     "19a2c0b92a6aa1580f1be2deb7b8a8e3a4857b6c6ccf522d00547878837267e7"
     "2ff9ac386eac4dffd77a33e93b0c8236bb376c5a5df62e36d4bfa821d56e4e20" default))
 '(display-line-numbers-type 'relative)
 '(ede-project-directories
   '("/home/edoardo/C-Projects/stui/include" "/home/edoardo/C-Projects/stui/src"
     "/home/edoardo/C-Projects/stui"))
 '(global-display-line-numbers-mode t)
 '(package-selected-packages
   '(ace-window auto-complete-clang-async cargo-mode company consult consult-dir
		consult-eglot consult-flycheck eglot-signature-eldoc-talkative
		eldoc exec-path-from-shell f gruber-darker-theme gruvbox-theme
		ht jsonrpc lsp-mode lsp-ui lv marginalia markdown-mode orderless
		rust-mode spinner vertico which-key zig-mode))
 '(select-enable-clipboard t)
 '(tool-bar-mode nil))

;; Set relative line number
(setq display-line-numbers 'relative)

;; Turn on visual bell
(setq visible-bell t)

;; set "gnu" style indenting for c
(setq c-default-style "linux"
      c-basic-offset 4)

;; Enable/Disable (t/nil) Emacs creating ~ backup files
(setq make-backup-files nil)

;; silently delete old backup
(setq delete-old-versions t)

;; Set a single directory to store all backup files
(setq backup-directory-alist '(("." . "~/.emacs.d/backup")))

;; MELPA packages
(require 'package)
(add-to-list 'package-archives '("melpa" . "https://melpa.org/packages/") t)
;; Comment/uncomment this line to enable MELPA Stable if desired.  See `package-archive-priorities`
;; and `package-pinned-packages`. Most users will not need or want to do this.
;; (add-to-list 'package-archives '("melpa-stable" . "https://stable.melpa.org/packages/") t)
(package-initialize)
(custom-set-faces
 ;; custom-set-faces was added by Custom.
 ;; If you edit it by hand, you could mess it up, so be careful.
 ;; Your init file should contain only one such instance.
 ;; If there is more than one, they won't work right.
 '(default ((t (:family "JetBrainsMonoNL Nerd Font Mono" :foundry "JB" :slant normal :weight semi-bold :height 142 :width normal)))))


(use-package marginalia
  :ensure t
  :config
  (marginalia-mode))


;; Enable vertico
(use-package vertico
  :init
  (vertico-mode)
  ;; Show more candidates
     (setq vertico-count 10)
     )

;; other things for vertico
;; Support opening new minibuffers from inside existing minibuffers.
  (setq enable-recursive-minibuffers t)

;; THIS IS THE BEST THING EVER
;; The `orderless' completion style.
(use-package orderless
  :init
  ;; Configure a custom style dispatcher (see the Consult wiki)
  ;; (setq orderless-style-dispatchers '(+orderless-consult-dispatch orderless-affix-dispatch)
  ;;       orderless-component-separator #'orderless-escapable-split-on-space)
  (setq completion-styles '(orderless basic)
        completion-category-defaults nil
        completion-category-overrides '((file (styles partial-completion)))))
(put 'downcase-region 'disabled nil)
(put 'upcase-region 'disabled nil)
