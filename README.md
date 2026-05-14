# Promptbook-request-evaluation

This repository contains 3 files to prepare, evaluate and revise codebooks for qualitative data analysis:

1.	Create Sample: Uses the Nebula LLM to translate the input text; Creates an excel sheet for manual annotation with the codes & samples.
2.	LLM coding: send the prompt & text to the LLM
3.	LLM evaluation: load the created files and manual annotation to calculate alphas and the confusion matrix; create an excel file for detailed fn & fp analysis.

To run it you need to add individual files to these folders:
The causes codebook.md file goes to: codebooks/ (used in create sample & LLM coding)
The manual annotation.xlsx files go to: MANUAL CODING/ (used for the LLM evaluation)
