import base64, os, torch
import streamlit as st
import numpy as np
from PIL import Image
from io import BytesIO

from neural_style.neural_style import stylize

#st.set_option("deprecation.showfileUploaderEncoding", False)
class Dict2Class(object):
    '''Turns a dictionary into a class'''
    def __init__(self, my_dict):
          
        for key in my_dict:
            setattr(self, key, my_dict[key])

def get_image_download_link(pil_im, str_msg = 'Download result',
        fname = None, str_format = 'JPEG'):
    """
    Generates a link allowing the PIL image to be downloaded
    in:  PIL image
    out: href string
    """
    buffered = BytesIO()
    pil_im.save(buffered, format= str_format, quality = 100)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    fname_str = f'download="{fname}"' if fname else ''
    href = f'<a href="data:file/jpg;base64,{img_str}" {fname_str}>{str_msg}</a>'
    return href

def get_models(model_dir, image_name = None, debug = False):
    assert os.path.isdir(model_dir), f"{model_dir} is not a valid path"
    m_list = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
    if debug:
        print(f'models found: \n {m_list}')
    m_list = [m for m in m_list if image_name in m] if image_name else m_list
    if debug:
        print(f'models matching {image_name}:\n{m_list}')
    return m_list

def get_styled_images(image_dir):
    assert os.path.isdir(image_dir), f"{model_dir} is not a valid path"
    im_list = [f for f in os.listdir(image_dir) if f.endswith(('.jpg','.png'))]
    return im_list

def downsize_im(pil_im, max_h = 500):
    o_w, o_h = pil_im.size
    h = min(o_h, max_h)
    w = int(o_w * h/ o_h)
    im_small = pil_im.resize((w,h), Image.ANTIALIAS) #best downsize filter
    return im_small

@st.cache
def style_image(pil_im, model_path, 
        use_cuda = torch.cuda.is_available(),
        tmp_img_path = '/tmp/pytorch_fnst.jpg'
        ):
    # Load Args Dictionary
    style_args = {
        'content_image': None,
        'pil_image': pil_im,
        'content_scale': None,
        'output_image': tmp_img_path,
        'model': model_path,
        'cuda': 1 if use_cuda else 0,
        'export_onnx': None
    }
    out_pil_im = stylize(Dict2Class(style_args))
    return out_pil_im

def Main():
    st.set_page_config(
        layout="wide",
        initial_sidebar_state="expanded",
    )

    url = "https://github.com/miroai/pytorch-fnst"
    st.markdown(f"# [PyTorch Fast Neural Style Transfer]({url}) demo")

    l_col , r_col = st.beta_columns(2)
    styled_im_dir = "styled_images"
    model_dir = "saved_models"

    with l_col:
        styled_im_name = st.selectbox(
            "Select Styled Image",
            options = [''] + sorted(get_styled_images(styled_im_dir))
        )
        if styled_im_name:
            styled_im_base, styled_im_ext = os.path.splitext(styled_im_name)
            styled_pil_im = downsize_im(Image.open(os.path.join(styled_im_dir, styled_im_name)), max_h = 300)
            st.image(styled_pil_im, styled_im_base)
            l_models = get_models(model_dir, image_name= styled_im_name, debug = False)
            l_intensity = [m.split('_')[-1].replace('.pth','') for m in l_models]
            assert len(l_models) == len(l_intensity), "number of models found must match number of intensity parsed"

            if len(l_intensity)>0:
                intensity = st.selectbox('style intensity', options = l_intensity)
                model_index = np.where(np.array(l_intensity) == intensity)[0][0]
                model_name = l_models[model_index]
            else:
                st.error(f'no models available for {styled_im_base}')
                return None
        else:
            return None
    with r_col:
        raw_image_bytes = st.file_uploader("Choose an image...", type = ['jpg', 'jpeg'], accept_multiple_files = False)
        use_cuda = st.checkbox('use CUDA') if torch.cuda.is_available() else False

    if raw_image_bytes is not None:
        im_name, im_ext = os.path.splitext(raw_image_bytes.name)
        out_im_name = im_name + f'_{styled_im_base}_{intensity}' + im_ext
        img0 = np.array(Image.open(raw_image_bytes))

        with st.spinner(text="Applying Style..."):
            in_pil_im = Image.fromarray(img0)
            max_in_im_h = r_col.number_input('max input image height (0 = no resize)', value = 500)
            in_pil_im = downsize_im(in_pil_im, max_h = max_in_im_h) if max_in_im_h else in_pil_im
                        
            out_pil_im = style_image(pil_im = in_pil_im, model_path = os.path.join(model_dir, model_name),
                            use_cuda= use_cuda)

        # Show Result
        l_col, r_col = st.beta_columns(2)
        with l_col:
            st.image(out_pil_im, caption = f"{styled_im_base} effects at {intensity} intensity")

            if st.checkbox('Download Image'):
                st.markdown(
                    get_image_download_link(out_pil_im,
                        str_msg = 'Click To Download Image',
                        fname = out_im_name),
                    unsafe_allow_html = True)
        with r_col:
            st.image(img0, caption = "Original")

if __name__ == '__main__':
    Main()
