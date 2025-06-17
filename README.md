## Các bước cài đặt

### 1. Tạo môi trường ảo Conda

Tạo và kích hoạt môi trường ảo để đảm bảo các thư viện được cài đặt trong một môi trường riêng biệt:

```bash
conda create --name myenv python=3.10 -y
conda activate myenv
```
Cài đặt các thư viện cần thiết từ file requirements.txt:

```bash
pip install -r requirements.txt
```
### 2. Thu thập và xử lý dữ liệu
Dữ liệu được thu thập từ các nguồn uy tín như: YouMed, AloBacsi, Bệnh viện Bạch Mai, Nhà thuốc Long Châu.    
Quy trình:  
Thu thập dữ liệu: Xem mã nguồn trong thư mục data/ để hiểu cách dữ liệu được crawl.  
Làm sạch và tiền xử lý: Dữ liệu được xử lý để đảm bảo định dạng phù hợp.  
Lưu trữ: Dataset đã được đẩy lên Hugging Face tại meandyou200175/dataset_full_fixed để lưu trữ lâu dài và tái sử dụng.
### 3. Tinh chỉnh mô hình
Mở terminal/command prompt và chạy lệnh sau để tinh chỉnh mô hình embedding: 
```bash
python finetune_embed.py
```
### 4. Tinh chỉnh mô hình ngôn ngữ lớn (LLMs)
Dự án sử dụng LLaMA-Factory để tinh chỉnh các mô hình ngôn ngữ lớn. Tham khảo tài liệu chính thức trên GitHub để biết thêm chi tiết về cách sử dụng và cấu hình.
### 5. Chạy ứng dụng
Để sử dụng model gpt-4o cho pipeline sử dụng câu lệnh:
```bash
bash run_openai.sh
```

 Sử dụng mô hình LLMs mã nguồn mở, ta cần khởi động trên vllm trước:
```bash
bash vllm.sh
```
Sau đó, chạy ứng dụng:
```bash
bash run_llama.sh
```
Ngoài ra bạn có thể sửa các file về prompt theo kỹ thuật prompt của riêng mình trong folder prompt
### 6. Các đánh giá
Để xem benchmark các mô hình embedding cho tiếng việt:
```bash
bench_mark.ipynb
```
Xem so sánh hiệu xuất của các mô hình sau khi đã finetune và áp dụng các kỹ thuật như hybrid search và rerank
```bash
bm25eval.ipynb # Đánh giá khả năng truy xuất của BM25
python eval_hybrid.py # So sánh hiệu xuất của các mô hình kết hợp hybrid search
python eval_ranking.py # So sánh hiệu xuất của các mô hình kết hợp hybrid search + rerank
```





















