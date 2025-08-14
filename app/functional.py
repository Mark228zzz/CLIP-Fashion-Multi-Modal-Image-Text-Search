import torch
import open_clip
from PIL import Image
import faiss
import pandas as pd

def main():
    global model, preprocess, tokenizer, device, val

    # Set the device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Download model and tokenizer
    model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-B-32', pretrained='laion2b_s34b_b79k', device=device
    )

    tokenizer = open_clip.get_tokenizer('ViT-B-32')

    # Load the model (if there is)
    ckpt = torch.load('./../models/clip_ft_best.pt', map_location=device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    # Read validation dataset
    val = pd.read_csv('./data/val.csv')

# Build HNSW index (for very fast search)
def build_hnsw(vectors, m=32, ef_search=64):
    dim = vectors.shape[1]
    index = faiss.IndexHNSWFlat(dim, m)  # m = neighbors per node
    index.hnsw.efSearch = ef_search
    index.add(vectors.astype('float32'))

    return index

# Encode all (with fine-tuned model)
@torch.no_grad()
def encode_all(df):
    img_embs, txt_embs = [], []

    for img_path, text in zip(df.image, df.text):
        # Preprocess images and tokenize texts
        img = preprocess(Image.open(img_path).convert('RGB')).unsqueeze(0).to(device)
        txt = tokenizer([text]).to(device)

        # Encode
        img_vec = model.encode_image(img)
        txt_vec = model.encode_text(txt)

        # Normalize
        img_vec /= img_vec.norm(dim=-1, keepdim=True)
        txt_vec /= txt_vec.norm(dim=-1, keepdim=True)

        # Append in a list
        img_embs.append(img_vec.cpu())
        txt_embs.append(txt_vec.cpu())

    return torch.cat(img_embs).numpy(), torch.cat(txt_embs).numpy()

# Encode all validation dataset
img_vectors, txt_vectors = encode_all(val)

# Create HNSW indeces for images and texts
index_img = build_hnsw(img_vectors)
index_txt = build_hnsw(txt_vectors)

# Search functions
def search_by_text(query, k=5):
    with torch.no_grad():
        # Tokenize and encode text
        q_tokens = tokenizer([query]).to(device)
        q_vec = model.encode_text(q_tokens)

        # Normalize
        q_vec /= q_vec.norm(dim=-1, keepdim=True)

        # Search
        D, I = index_img.search(q_vec.cpu().numpy().astype('float32'), k)

    results = []
    for score, idx in zip(D[0], I[0]):
        # Take best images' paths by ids
        img_path = val.image.iloc[idx]

        try:
            # Open an image, make a caption and append in list best results
            img_obj = Image.open(img_path).convert('RGB')
            caption = f"{val.text.iloc[idx]} (score={score:.3f})"
            results.append((img_obj, caption))
        except:
            continue

    return results

def search_by_image(img, k=5):
    with torch.no_grad():
        # Preprocess and encode image
        img_tensor = preprocess(img).unsqueeze(0).to(device)
        q_vec = model.encode_image(img_tensor)

        # Normalize
        q_vec /= q_vec.norm(dim=-1, keepdim=True)

        # Search
        D, I = index_txt.search(q_vec.cpu().numpy().astype('float32'), k)

    captions = [f"{val.text.iloc[idx]} (score={score:.3f})" for score, idx in zip(D[0], I[0])]

    return "\n".join(captions)

# Only runned as a lib
if __name__ != '__main__':
    main()
