import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

import streamlit as st
from PIL import Image

import io
from datetime import datetime
from fpdf import FPDF

from model_utils import build_cnn


st.set_page_config(page_title='Brain Tumor MRI', layout='wide')

ROOT = Path(__file__).parent
DATA_DIR = ROOT / 'data'
MODEL_DIR = ROOT / 'model'
CLASSES = json.load(open(MODEL_DIR / 'class_names.json'))
IMG_EXT = {'.jpg', '.jpeg', '.png'}


@st.cache_data
def list_images(split, cls):
    folder = DATA_DIR / split / cls
    return sorted(p for p in folder.glob('*') if p.suffix.lower() in IMG_EXT)


# ---------- Page a: data exploration ----------
def page_exploration():
    st.header('Data exploration')
    c1, c2 = st.columns(2)
    split = c1.selectbox('Split', ['Training', 'Testing'])
    cls = c2.selectbox('Class', CLASSES)

    files = list_images(split, cls)
    if not files:
        st.error(f'No images found in {DATA_DIR / split / cls}')
        return
    st.caption(f'{len(files)} images')

    per_page = 12
    n_pages = math.ceil(len(files) / per_page)
    page = st.number_input('Page', 1, n_pages, 1)
    subset = files[(page - 1) * per_page: page * per_page]

    cols = st.columns(4)
    for i, p in enumerate(subset):
        cols[i % 4].image(str(p), caption=p.name, use_container_width=True)

    st.divider()
    st.subheader('Enlarge an image')
    name = st.selectbox('Select an image (from the current page)', [p.name for p in subset])
    path = next(p for p in subset if p.name == name)
    img = Image.open(path)
    st.image(img, caption=f'{cls} | {img.size[0]}x{img.size[1]} px', width=600)


# ---------- Placeholders (next steps) ----------
# ---------- Cached loaders ----------
@st.cache_resource
def load_model():
    params = json.load(open(MODEL_DIR / 'best_params.json'))
    model = build_cnn(params['base_filters'], params['dropout'])
    model.load_weights(MODEL_DIR / 'best.weights.h5')
    return model

@st.cache_data
def load_json(name):
    return json.load(open(MODEL_DIR / name))


def model_summary_text(model):
    lines = []
    model.summary(print_fn=lambda x, **kw: lines.append(x))
    return '\n'.join(lines)


def plot_confusion_matrix(cm, classes):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(len(classes)), classes, rotation=45)
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i][j], ha='center', va='center',
                    color='white' if cm[i][j] > np.max(cm) / 2 else 'black')
    fig.tight_layout()
    return fig


def plot_curves(hist, title):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for ax, m in zip(axes, ['accuracy', 'loss']):
        ax.plot(hist[m], label='train')
        ax.plot(hist['val_' + m], label='test')
        ax.set_title(f'{title} - {m}'); ax.set_xlabel('epoch'); ax.legend()
    fig.tight_layout()
    return fig


# ---------- Page b: model ----------
def page_model():
    st.header('Model')
    model = load_model()
    metrics = load_json('best_metrics.json')
    params = load_json('best_params.json')

    st.subheader('Parameters')
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('**Hyperparameters (Optuna best)**')
        st.table(pd.DataFrame({'value': params}).astype(str))
    with c2:
        st.markdown('**Architecture**')
        st.code(model_summary_text(model), language='text')

    st.subheader('Test metrics')
    m1, m2 = st.columns(2)
    m1.metric('Test accuracy', f"{metrics['test_accuracy']:.4f}")
    m2.metric('Test loss', f"{metrics['test_loss']:.4f}")

    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown('**Classification report**')
        rep = pd.DataFrame(metrics['report']).T.round(3)
        st.dataframe(rep, use_container_width=True)
    with c2:
        st.markdown('**Confusion matrix**')
        st.pyplot(plot_confusion_matrix(metrics['confusion_matrix'], CLASSES))

    st.subheader('Learning curves')
    st.pyplot(plot_curves(load_json('best_history.json'), 'Optimized'))
    if st.checkbox('Compare with baseline'):
        st.pyplot(plot_curves(load_json('baseline_history.json'), 'Baseline'))
        
def preprocess(img):
    # Same as training: RGB, 150x150, rescale to [0,1].
    # NEAREST matches Keras' flow_from_directory default interpolation.
    img = img.convert('RGB').resize((150, 150), Image.NEAREST)
    return np.asarray(img, dtype='float32')[None] / 255.0


# ---------- Grad-CAM (bonus) ----------
def gradcam(model, x, layer_name='gradcam_conv', class_idx=None):
    """Heatmap (12x12, values in [0,1]) of the regions that drove the prediction."""
    x = tf.convert_to_tensor(x)
    with tf.GradientTape() as tape:
        h, conv_out = x, None
        for layer in model.layers:          # manual forward pass, no sub-model needed
            h = layer(h)
            if layer.name == layer_name:
                conv_out = h
                tape.watch(conv_out)
        if class_idx is None:
            class_idx = int(tf.argmax(h[0]))
        score = h[:, class_idx]
    grads = tape.gradient(score, conv_out)
    weights = tf.reduce_mean(grads, axis=(1, 2))                       # one weight per filter
    cam = tf.reduce_sum(conv_out * weights[:, None, None, :], axis=-1)[0]
    cam = tf.nn.relu(cam).numpy()
    return cam / (cam.max() + 1e-8), class_idx


def overlay_heatmap(img, cam, alpha=0.4):
    cam_img = Image.fromarray(np.uint8(255 * cam)).resize(img.size, Image.BILINEAR)
    colored = plt.get_cmap('jet')(np.asarray(cam_img) / 255.0)[..., :3]
    colored = Image.fromarray((colored * 255).astype('uint8'))
    return Image.blend(img.convert('RGB'), colored, alpha)

# ---------- Page c: test on an image ----------
def page_test():
    st.header('Test on an image')
    source = st.radio('Image source', ['Dataset (Testing)', 'Upload'], horizontal=True)

    true_label = None
    if source == 'Dataset (Testing)':
        c1, c2 = st.columns(2)
        cls = c1.selectbox('Class folder', CLASSES)
        files = list_images('Testing', cls)
        name = c2.selectbox('Image', [p.name for p in files])
        img = Image.open(next(p for p in files if p.name == name))
        true_label = cls
    else:
        up = st.file_uploader('Upload an MRI image', type=['jpg', 'jpeg', 'png'])
        if up is None:
            return
        img = Image.open(up)

    img = img.convert('RGB')
    model = load_model()
    probs = model.predict(preprocess(img), verbose=0)[0]
    pred = CLASSES[int(probs.argmax())]

    # Kept for the PDF report and Grad-CAM
    cam, _ = gradcam(model, preprocess(img))
    heat = overlay_heatmap(img, cam)

    # Kept for the PDF report
    st.session_state['last_result'] = dict(image=img, probs=probs, pred=pred,
                                           true=true_label, heatmap=heat)
    c1, c2 = st.columns(2)
    c1.image(img, caption='Input image', use_container_width=True)
    with c2:
        st.subheader(f'Prediction: {pred}')
        st.metric('Confidence', f'{probs.max():.1%}')
        if true_label:
            st.write('True class:', true_label, '✅' if pred == true_label else '❌')
        df = pd.DataFrame({'probability': probs}, index=CLASSES)
        st.bar_chart(df)
        st.dataframe(df.style.format('{:.2%}'), use_container_width=True)
        
    st.subheader('Interpretability (Grad-CAM)')
    g1, g2 = st.columns(2)
    g1.image(img, caption='Original', use_container_width=True)
    g2.image(heat, caption=f'Grad-CAM for predicted class: {pred}', use_container_width=True)
    st.caption('Red = regions that most increased the score of the predicted class.')
        
        
# ---------- PDF report ----------
def build_pdf(result, metrics, params, model, heatmap=None):
    pdf = FPDF()
    pdf.set_auto_page_break(True, 15)
    pdf.add_page()

    def title(text, size=12):
        pdf.ln(3)
        pdf.set_font('Helvetica', 'B', size)
        pdf.cell(0, 8, text, new_x='LMARGIN', new_y='NEXT')
        pdf.set_font('Helvetica', '', 10)

    def table(rows, widths, header=True):
        for i, row in enumerate(rows):
            pdf.set_font('Helvetica', 'B' if (header and i == 0) else '', 10)
            for val, w in zip(row, widths):
                pdf.cell(w, 7, str(val), border=1)
            pdf.ln()
        pdf.set_font('Helvetica', '', 10)

    def add_image(img, w=60):
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        buf.seek(0)
        pdf.image(buf, w=w)

    pdf.set_font('Helvetica', 'B', 16)
    pdf.cell(0, 10, 'Brain Tumor MRI - Classification Report', align='C', new_x='LMARGIN', new_y='NEXT')
    pdf.set_font('Helvetica', '', 9)
    pdf.cell(0, 6, datetime.now().strftime('%Y-%m-%d %H:%M'), align='C', new_x='LMARGIN', new_y='NEXT')

    title('1. Tested image')
    add_image(result['image'])
    pdf.ln(2)
    pdf.cell(0, 7, f"Predicted class: {result['pred']} ({result['probs'].max():.1%})",
             new_x='LMARGIN', new_y='NEXT')
    if result.get('true'):
        pdf.cell(0, 7, f"True class: {result['true']}", new_x='LMARGIN', new_y='NEXT')

    if heatmap is not None:  # Grad-CAM (bonus), added later
        title('Grad-CAM heatmap')
        add_image(heatmap)

    title('2. Prediction probabilities')
    table([['Class', 'Probability']] +
          [[c, f'{p:.2%}'] for c, p in zip(CLASSES, result['probs'])], [60, 40])

    title('3. Model metrics (test set)')
    pdf.cell(0, 7, f"Accuracy: {metrics['test_accuracy']:.4f}   Loss: {metrics['test_loss']:.4f}",
             new_x='LMARGIN', new_y='NEXT')
    rep = metrics['report']
    table([['Class', 'Precision', 'Recall', 'F1', 'Support']] +
          [[c, f"{rep[c]['precision']:.3f}", f"{rep[c]['recall']:.3f}",
            f"{rep[c]['f1-score']:.3f}", int(rep[c]['support'])] for c in CLASSES],
          [40, 30, 30, 30, 30])
    pdf.ln(3)
    pdf.cell(0, 7, 'Confusion matrix (rows = true, columns = predicted)', new_x='LMARGIN', new_y='NEXT')
    table([['True \\ Pred'] + CLASSES] +
          [[c] + list(row) for c, row in zip(CLASSES, metrics['confusion_matrix'])],
          [40, 35, 35, 35, 35])

    title('4. Model parameters')
    f = params['base_filters']
    table([['Parameter', 'Value'],
           ['Input', '150x150x3 (RGB, rescaled to [0,1])'],
           ['Conv layers (4x4, ReLU)', f'{f}, {2*f}, {4*f}, {4*f} filters'],
           ['Pooling', 'MaxPooling 3x3 (after conv 1-3)'],
           ['Dense / Dropout', f"512 / {params['dropout']}"],
           ['Output', 'Dense 4, softmax'],
           ['Optimizer', 'Adam'],
           ['Learning rate', f"{params['lr']:.6f}"],
           ['Batch size', params['batch_size']],
           ['beta_1 / beta_2', f"{params['beta_1']:.3f} / {params['beta_2']:.3f}"],
           ['Total parameters', f'{model.count_params():,}']],
          [60, 110])
    pdf.ln(4)
    pdf.set_font('Helvetica', 'I', 8)
    pdf.multi_cell(0, 5, 'Educational project. This output is not a medical diagnosis '
                         'and must be reviewed by a qualified practitioner.')
    return bytes(pdf.output())


# ---------- Page d: PDF report ----------
def page_report():
    st.header('PDF report')
    result = st.session_state.get('last_result')
    if result is None:
        st.info('Run a prediction in "Test an image" first.')
        return

    c1, c2 = st.columns([1, 2])
    c1.image(result['image'], caption='Image in the report', use_container_width=True)
    c2.write(f"Prediction: **{result['pred']}** ({result['probs'].max():.1%})")

    pdf_bytes = build_pdf(result, load_json('best_metrics.json'),
                          load_json('best_params.json'), load_model(),
                          heatmap=result.get('heatmap'))
    
    st.download_button('Download PDF report', pdf_bytes,
                       file_name='brain_tumor_report.pdf', mime='application/pdf')

# ---------- Navigation ----------
PAGES = {
    'Data exploration': page_exploration,
    'Model': page_model,
    'Test an image': page_test,
    'PDF report': page_report,
}
choice = st.sidebar.radio('Navigation', list(PAGES))
PAGES[choice]()