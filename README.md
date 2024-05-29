# Assignment 3 : Lay Summarization of Biomedical Research Articles
### Goal
- We participated in an ongoing global competition BioLaysumm
- The goal was to use a pre-trained language model (PLM) and train a system that produces
a layman’s summary given a research publication from the biomedical domain
- Given the abstract and main text of an article, your goal is to train a PLM-based model
that generates a layman’s summary for the article
- We were allowed to start with small or base versions of the Flan-T5 and BioGPT to ensure fairness
- Our models were evaluated on the following metrics 
  - Relevance: ROUGE (1, 2, and L) and BERTScore 
  - Readability: Flesch-Kincaid Grade Level (FKGL)
  - Factuality - AlignScore

### Idea Overview
- You can find the final submission in final_submission_folder and find the data on the competition website (https://www.codabench.org/competitions/1920/)
- Also you would need HPC even to just run the evalusation script as it uses good amount of GPU for the evaluation as well
- We started with just prompt engineering on both Flan T5 and BioGPT just to see an initial
performance for both the models without any fine-tuning
- We could clearly see that Flan T5 was outperforming BioGPT by a margin and hence we
shifted our attention towards fine-tuning Flan T5
- We referred to some articles on how to start with fine-tuning pre-trained models and also
referred to some papers on some tricks for laymann summarization
- We first merged both the datasets and shuffled them
- After loading pre-trained Flan T5 we tried to use LoRA but we were facing some version
mismatch issues which we tried to resolve for a while but due to time crunch we decided
to go with normal fine tuning the complete model
- We then started with the small model as we thought we won’t be able to train base and
large models in the given time frame but as we trained we found some bugs in the code
and after rectifying them we were able to completely train Flan T5-base
- The prompt that we were using while fine-tuning started like this "Write the highlights of
this article for a layman:" but we think that it could have been better if we used something
like "Summarize this for a layman"
- Number of epochs on which we trained was 4 as on trying with 6 we could see catastrophic
forgetting as we could see a decrease in the scores for the summaries generated

### Results
We finally secured a global rank 8 in the competition till the deadline of the assignment and stood 2nd in the class with a ROUGE score 0.4396

Please let me know if you have something to ask in the code :)

It could be a good starting point for learning how to fine tune LLMs