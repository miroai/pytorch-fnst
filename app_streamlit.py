import base64, os, torch
import streamlit as st
import numpy as np
from PIL import Image
from io import BytesIO

from neural_style.neural_style import stylize

#st.set_option("deprecation.showfileUploaderEncoding", False)
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

def get_models(model_dir, image_name = None):
	assert os.path.isdir(model_dir), f"{model_dir} is not a valid path"
	m_list = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
	m_list = [m for m in m_list if image_name in m] if image_name else m_list
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
	tmp_image_path = '/tmp/pytorch_fnst.jpg'

	with l_col:
		styled_im_name = st.selectbox(
			"Select Styled Image",
			options = [''] + sorted(get_styled_images(styled_im_dir))
		)
		if styled_im_name:
			styled_im_base, styled_im_ext = os.path.splitext(styled_im_name)
			styled_pil_im = downsize_im(Image.open(os.path.join(styled_im_dir, styled_im_name)))
			st.image(styled_pil_im, styled_im_base)
			l_models = get_models(model_dir, image_name= styled_im_name)
			l_intensity = [m.split('_')[-1].replace('.pth','') for m in l_models]
			assert len(l_models) == len(l_intensity), "number of models found must match number of intensity parsed"

			if len(l_intensity)> 0:
				st.error('no models available for {styled_im_base}')
			else:
				intensity = st.selectbox('style intensity', options = l_intensity)
				model_index = np.where(np.array(l_intensity) == intensity)[0]
				model_name = l_models[model_index]
		else:
			return None
	with r_col:
		raw_image_bytes = st.file_uploader("Choose an image...", type = ['jpg', 'jpeg'], accept_multiple_files = False)

	if raw_image_bytes is not None:
		im_name, im_ext = os.path.splitext(raw_image_bytes.name)
		out_im_name = im_name + f'_{styled_im_base}_{intensity}' + im_ext
		img0 = np.array(Image.open(raw_image_bytes))

		with st.spinner(text="Applying Style..."):
			# Load Args Dictionary
			style_args = {
				'content-image': None,
				'pil_image': Image.fromarray(img0),
				'content-scale': None,
				'output-image': tmp_image_path,
				'model': os.path.join(model_dir, model_name),
				'cuda': 1 if torch.cuda.is_available else 0,
				'export_onnx': None
			}
			# Apply the model, convert to BGR first and after
			out_pil_im = stylize(style_args)

		# Show Result
		l_col, r_col = st.beta_columns(2)
		with l_col:
			st.image(out_pil_im, caption = f"{styled_im_name} effects at {intensity} intensity")

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
