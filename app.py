import gradio as gr
from inference import translate

CSS = '''
.gradio-container {background:linear-gradient(130deg,#fff1a6 0%,#ffd6eb 46%,#b8edff 100%)!important;min-height:100vh!important;font-family:Inter,ui-sans-serif,system-ui!important}
#hero {text-align:center;margin:18px auto 26px!important}
#hero h1 {font-size:clamp(36px,6vw,64px)!important;font-weight:850!important;color:#332457!important;letter-spacing:-2px}
#hero p {color:#655778!important;font-size:17px!important}
#main-card {background:rgba(255,255,255,.91)!important;border:1px solid #fff!important;border-radius:26px!important;padding:28px!important;box-shadow:0 18px 50px rgba(100,58,137,.13)!important;max-width:800px!important;margin:auto!important}
#translate-btn {background:linear-gradient(90deg,#ff5a9e,#8c66fa)!important;color:white!important;border:none!important;font-weight:800!important;font-size:18px!important;border-radius:15px!important;min-height:54px!important}
#translate-btn:hover {filter:brightness(1.07)!important}
#subtitle {text-align:center;color:#6d6282!important;margin:24px auto!important}
footer {display:none!important}
'''
with gr.Blocks(title='LingoUz | English → Uzbek', css=CSS, theme=gr.themes.Soft(primary_hue='pink', secondary_hue='purple')) as demo:
    gr.HTML('<div id="hero"><h1>🌈 LingoUz ✨</h1><p>English → Uzbek • A tiny Transformer trained from scratch</p></div>')
    with gr.Column(elem_id='main-card'):
        gr.Markdown('### 💬 Translate a word')
        english = gr.Textbox(label='🇬🇧 English', placeholder='Type a word, e.g. hello', lines=1)
        translate_button = gr.Button('✨ Translate now', elem_id='translate-btn', variant='primary')
        uzbek = gr.Textbox(label='🇺🇿 Uzbek', interactive=False, lines=1, placeholder='Your translation will appear here')
        gr.Examples(examples=['hello','book','water','mother','school','computer','friend','apple'], inputs=english)
        translate_button.click(fn=translate, inputs=english, outputs=uzbek)
        english.submit(fn=translate, inputs=english, outputs=uzbek)
    gr.HTML('<div id="subtitle">🧠 2017 Transformer architecture · 🔤 Character-level · 📚 100 training pairs<br>Educational demo — translations outside the training set may be unreliable.</div>')
if __name__ == '__main__':
    demo.launch()
