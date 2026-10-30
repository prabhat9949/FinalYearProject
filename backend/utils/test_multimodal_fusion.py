from multimodal_fusion import generate_multimodal_score


result = generate_multimodal_score(
    clinical_probability=0.95,
    byra_steatosis_probability=0.006,
    behsof_nafld_probability=0.591
)

print(result)