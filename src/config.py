import torch 

class CONFIG:
    gen_dir = '/Users/armandbryan/Documents/aivancity/PGE5/NLP/NLP_AIVANCITY_COURSE/GeneratedTextDetection-main/Dataset/FullyGenerated'
    hybrid_dir = '/Users/armandbryan/Documents/aivancity/PGE5/NLP/NLP_AIVANCITY_COURSE/GeneratedTextDetection-main/Dataset/Hybrid_AbstractDataset'
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")


