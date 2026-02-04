# config.py
SELECTED_FEATURES = ['h1n1_concern', 'h1n1_knowledge', 'behavioral_antiviral_meds',
       'behavioral_avoidance', 'behavioral_face_mask', 'behavioral_wash_hands',
       'behavioral_large_gatherings', 'behavioral_outside_home',
       'behavioral_touch_face', 'doctor_recc_h1n1', 'doctor_recc_seasonal',
       'chronic_med_condition', 'child_under_6_months', 'health_worker',
       'opinion_h1n1_vacc_effective', 'opinion_h1n1_risk',
       'opinion_h1n1_sick_from_vacc', 'opinion_seas_vacc_effective',
       'opinion_seas_risk', 'opinion_seas_sick_from_vacc', 'age_group',
       'education', 'race', 'sex', 'income_poverty', 'marital_status',
       'rent_or_own', 'employment_status', 'census_msa', 'household_adults',
       'household_children']


TARGETS = ["h1n1_vaccine", "seasonal_vaccine"]

MAPPING = {
  "age_group": {
    "65+ Years": 1,
    "55 - 64 Years": 2,
    "45 - 54 Years": 3,
    "18 - 34 Years": 4,
    "35 - 44 Years": 5,
    "Other": 6
  },
  "education": {
    "College Graduate": 1,
    "Some College": 2,
    "12 Years": 3,
    "Other": 4
  },
  "race": {
    "White": 1,
    "Black": 2,
    "Hispanic": 3,
    "Other": 4
  },
  "income_poverty": {
    "<= $75,000, Above Poverty": 1,
    "> $75,000": 2,
    "Below Poverty": 3
  },
  "marital_status": {
    "Married": 1,
    "Not Married": 2
  },
  "rent_or_own": {
    "Own": 1,
    "Rent": 2
  },
  "sex": {
    "Female": 1,
    "Male": 2
  },
  "employment_status": {
    "Not in Labor Force": 1,
    "Employed": 2,
    "Unemployed": 3
  },
  "census_msa": {
    "MSA, Not Principle  City": 1,
    "MSA, Principle City": 2,
    "Non-MSA": 3
  }
}
