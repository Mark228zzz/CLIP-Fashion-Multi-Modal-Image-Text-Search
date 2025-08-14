import gradio as gr
from .functional import search_by_image, search_by_text

with gr.Blocks() as demo:
    gr.Markdown('## 🖼️ Text ↔ Image Search (CLIP + FAISS HNSW)')

    with gr.Tab('Text → Images'):
        with gr.Row():
            text_in = gr.Textbox(label='Search text', scale=3)
            k_in = gr.Slider(1, 20, value=5, step=1, label='Top‑K', scale=1)

        gallery_out = gr.Gallery(label='Top Matches', columns=5, height='auto')
        dl_btn = gr.Button('Download Top‑K as ZIP')
        dl_file = gr.File(label='Your ZIP will appear here')

        # search on Enter
        text_in.submit(fn=search_by_text, inputs=[text_in, k_in], outputs=gallery_out)
        # also allow click to search
        k_in.change(fn=search_by_text, inputs=[text_in, k_in], outputs=gallery_out)

    with gr.Tab('Image → Text'):
        img_in = gr.Image(label='Upload image', type='pil')
        k2_in = gr.Slider(1, 20, value=5, step=1, label='Top‑K')
        text_out = gr.Textbox(label='Top Captions + Scores')
        img_in.change(fn=search_by_image, inputs=[img_in, k2_in], outputs=text_out)

# Only runned directly
if __name__ == '__main__':
    demo.launch(share=True)
