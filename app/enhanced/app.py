import streamlit as st
from PIL import Image
import torch
from config import BaseConfig, current_config, ModelConfig
from utils.producer import LatexProducer
from utils.vocab import Vocab
from models.CNN_LSTM import CNN_LSTM
from models.ViT_LSTM import ViT_LSTM
import pickle as pkl
from utils.transform import transform, visual_transform
import numpy as np
from utils.visuallize import visualize_attention



st.set_page_config(page_title="Image to LaTeX Demo", layout="wide")

# --- SIDEBAR: Cấu hình ---
st.sidebar.title("⚙️ Cấu hình hệ thống")
selected_model_name = st.sidebar.selectbox("Chọn Model Weight", list(BaseConfig.MODELS.keys()))
current_config["beam_size"] = st.sidebar.slider("Beam Size", 1, 10, 5)
current_config["max_len"] = st.sidebar.number_input("Max Length", value=150)

# --- LOAD MODEL (Caching để tránh load lại mỗi lần render) ---
@st.cache_resource
def load_model_and_producer(model_name):
    weight_path = BaseConfig.MODELS[model_name]
    # 1. Load Vocab 
    with open(current_config["vocab_path"], "rb") as f:
        vocab = pkl.load(f)
    vocab_size = len(vocab)
    # 2. Khởi tạo Model & Load Weight
    if model_name == "CNN & LSTM":
        model = CNN_LSTM(vocab_size, ModelConfig.CONFIG["word_emb_dim"], ModelConfig.CONFIG["rnn_h_dim"], ModelConfig.CONFIG["rnn_o_dim"], ModelConfig.CONFIG["enc_out_dim"], ModelConfig.CONFIG["attn_dim"], ModelConfig.CONFIG["n_heads"], 0)
    else:
        model = ViT_LSTM(vocab_size, ModelConfig.CONFIG["word_emb_dim"], ModelConfig.CONFIG["rnn_h_dim"], ModelConfig.CONFIG["rnn_o_dim"], ModelConfig.CONFIG["enc_out_dim"], ModelConfig.CONFIG["attn_dim"], ModelConfig.CONFIG["n_heads"], 0)
    model.load_state_dict(torch.load(weight_path, map_location='cpu'))
    # 3. Khởi tạo Producer
    producer = LatexProducer(model, vocab, max_len=current_config["max_len"], use_cuda=BaseConfig.DEFAULT_CONFIG["use_cuda"], beam_size=current_config["beam_size"])
    return producer

producer = load_model_and_producer(selected_model_name)

# --- MAIN UI ---
st.title("📸 Image to LaTeX OCR")
st.markdown("Tải ảnh công thức toán học lên để chuyển đổi sang mã LaTeX.")

uploaded_file = st.sidebar.file_uploader("Chọn ảnh công thức...", type=["jpg", "jpeg", "png"])

col1, col2 = st.columns(2)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    
    with col1:
        st.subheader("Ảnh đầu vào")
        st.image(image, use_container_width=True)
    
    if st.button("🚀 Chuyển đổi (Predict)"):
        with st.spinner("Đang xử lý..."):
            img_np = np.array(image)

            # 1. Chuyển đổi sang tensor để predict
            img_dict = transform(image=img_np)
            img_tensor = img_dict['image'].unsqueeze(0) # [1, C, H, W]
            
            # 2. Logic decode (Đảm bảo LatexProducer đã được sửa để trả về alphas như hướng dẫn trước)
            result, alphas = producer(img_tensor)
            
            # 3. Tính toán H_prime, W_prime (Kích thước feature map cuối của Encoder)
            _, H_feat, W_feat = img_dict['image'].shape
            H_prime, W_prime = H_feat // 8, W_feat // 8 
            
            # 5. Gọi hàm visualize
            st.subheader("🔍 Attention Analysis")
            
            # Giả sử hàm visualize_attention của bạn trả về một Figure của Matplotlib
            fig = visualize_attention(
                visual_transform(image=img_np)['image'], 
                result[0].split(), 
                alphas[0], 
                H_prime, 
                W_prime
            )
            
            # HIỂN THỊ LÊN STREAMLIT
            if fig is not None:
                st.pyplot(fig)
            
            # Lưu kết quả vào session_state
            st.session_state['latex_code'] = result[0]

# --- HIỂN THỊ KẾT QUẢ ---
if 'latex_code' in st.session_state:
    with col2:
        st.subheader("Kết quả LaTeX")
        
        # Cho phép người dùng sửa trực tiếp
        edited_latex = st.text_area("Mã LaTeX (Bạn có thể sửa lỗi tại đây):", 
                                    value=st.session_state['latex_code'], 
                                    height=150)
        
        # Nút Copy (Streamlit mặc định hỗ trợ copy trong text_area hoặc code block)
        st.code(edited_latex, language='latex')
        
        # Render ngược lại để kiểm chứng
        st.subheader("Bản xem trước (Rendered)")
        try:
            st.latex(edited_latex)
        except Exception as e:
            st.error(f"Lỗi render LaTeX: {e}")

    st.success("Hoàn thành! Bạn có thể copy mã LaTeX phía trên.")