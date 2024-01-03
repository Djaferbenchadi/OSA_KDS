# Malware Visualization based on OSA_KDS


This project come as the second part of our [IEEE access paper](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=JQsEaPUAAAAJ&citation_for_view=JQsEaPUAAAAJ:IjCSPb-OGe4C), following the [malware classification task](https://github.com/Djaferbenchadi/Malware_classification_ksm). 

Here we introduce a visualization framework named "Occlusion Sensitivity Analysis based Kernel Difference Subspace (OSA_KDS)". 

This wor focuses on understanding malware behavior. The framework is essential for enhancing the interpretability of malware detection by generating saliency maps to visualize the importance of discriminative features within different malware families.
OSA_KDS measures the significance of specific elements in a feature vector that discriminates between malware and safe classes. This measurement is based on the change in the vector's length when an element is occluded by a small window mask. The discriminative feature vector is extracted by projecting a malware pattern vector onto a Kernel Difference Subspace (KDS), representing the differences between malware and safe class subspaces. By sliding the mask across the feature vector, we determine each element's importance, resulting in a saliency map that visualizes these crucial elements.


<img src="https://github.com/Djaferbenchadi/OSA_KDS/blob/main/OSA-KDS-diag.png" />

## Future work 


## Citation

If you are using OSA_KDS in an academic project, we would be grateful if you could reference our work using the following BibTeX entry:


```
@article{djafer2023efficient,
  title={Efficient Malware Analysis Using Subspace-Based Methods on Representative Image Patterns},
  author={Djafer Yahia M, Benchadi and Batalo, Bojan and Fukui, Kazuhiro},
  journal={IEEE Access},
  volume={11},
  pages={102492--102507},
  year={2023}
}
```

Computer Vision Lab (CVLAB), University of Tsukuba, Japan
