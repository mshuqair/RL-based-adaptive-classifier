# Deep Reinforcement-Learning-Based Adaptive Classifier
This unique approach uses Reinforcement Learning (RL) to discern shifts in data stream distributions during state transitions.
Training an RL agent to recognize these transitions makes it adept at identifying transitions in new data.
Instead of static models, our agent interacts with the data's dynamics and makes optimal classification decisions.
This RL-driven framework prioritizes understanding changes in data distribution, making it robust against inter and intra-data variations.

![](figures/figure_rl_structure.png)
**Figure 1.** The proposed reinforcement-learning-based adaptive classification framework.

![](output/mhealth_p1_model_output.png)
**Figure 2.** The model prediction for Walking vs. Non-walking (1 vs. 2) in one participant of the MHEALTH dataset.

## Updates
More updates regarding the description are coming soon.
Code updates:
 - Update the model train and test to improve readability and reproducibility
 - Updated to Python 3.10.11
 - Updated to TensorFlow 2.10.0
 - Various updates to plotting functions


## Deep Reinforcement Learning Adaptive Classification of PD Medication State
- The preliminary results of this project were published at the IEEE ICDM 2022 Conference.
Incremental Learning in Time-series Data using Reinforcement Learning: https://doi.org/10.1109/ICDMW58026.2022.00115
- The IEEE Journal of Biomedical and Health Informatics has recently published an extensive extension of this work.
Reinforcement Learning-Based Adaptive Classification for Medication State Monitoring in Parkinson's Disease: https://doi.org/10.1109/JBHI.2024.3423708
- Please cite the papers if you find this work useful


## Code Requirements and Compatability
The code was run and tested using the following:
- Python			3.10.11
- tensorflow		2.10.1
- keras				2.10.0
- matplotlib		3.10.1
- numpy				1.26.3
- pandas			2.2.3
- scikit-learn		1.6.1


## Conclusions

