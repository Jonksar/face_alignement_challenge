# Mark non-file targets as PHONY
.PHONY: all beautify

beautify:
	black -t py36 -l 120 .
