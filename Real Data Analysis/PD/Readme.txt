Periodontal Disease Data

# This folder contains three files:
- README.txt
- code.R: The code to load data
- PDdata.csv: Periodontal disease (PD) is a collection of chronic inflammatory diseases caused by bacterial infection of the supporting 
	      tissues or periodontium around the teeth. We consider GAAD (Gullah African-American Diabetics) data from a clinical study 
	      of PD (Fernandes et al. 2009). Correlated periodontal pocket depth (PPD), and clinical attachment level (CAL), measured at six pre-specified
	      sites for each tooth by hygienists. The total number of teeth is 28. Five important covariates are recorded, including age, 
	      gender, body mass index (BMI), smoking status and hemoglobin A1c(HbA1c). The data set can be summarized as a sites × teeth × P P D&CAL 
	      = 6 × 28 × 2 tensor response and a 5 dimensional predictor. Some values of PPD and CAL are missing for the data set,
	      we select 153 patients with at least 70% of the response observed, then we use the predictive mean matching method provided 
	      in R package “mice” to complete the data.

