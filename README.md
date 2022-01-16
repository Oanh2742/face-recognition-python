# face-recognition-python
0. Các package cần cài đặt trước khi chạy
$ pip install -r requirements.txt
1. Cần có 2 file: known_faces.pth và known_names.npy có chứa dữ liệu (khác rỗng) để đánh giá trong quá trình nhận diện gương mặt.
  Trong đó: 
    + known_faces.pth chứa các feature-vector của gương mặt đã biết 
    + known_names.npy tên người tương ứng
  Để tạo ra được 2 file trên cần chạy lệnh sau:
$ python update_database.py
  Khi đó, known_faces.pth và known_names.npy được tạo ra dựa trên dữ liệu gương mặt hiện có trong folder data/base_images
2. Để lấy được ảnh gương mặt chính xác, đúng kích cỡ và đưa vào data/base_images, cần:
  - Cách 1: Trích xuất từ ảnh
    + Đưa ảnh có mặt người vào folder raw
    + Chạy lệnh: $ python register_face_image.py
    + Nhập vào tên file ảnh
    + Ảnh các gương mặt detect được sẽ hiện ra cùng với số thứ tự
    + Nhập tên người đúng theo số thứ tự, nếu không biết thì nhập "u" hoặc "unknown"
   - Cách 2: Dùng webcam đăng ký trực tiếp
    + Chạy lệnh: $ python register_face_webcam.py
    + Giữ gương mặt chỉ một người trước webcam một lúc
    + Nhập tên người đó
   Khi đó (các) folder mang tên người được tạo ra trong data/base_images nếu chưa tồn tại folder đó, 
   và (các) ảnh gương mặt được cắt chỉnh đúng kích cỡ được đưa vào đúng folder mang tên mỗi người.
  3. Nhận diện gương mặt 
  - Kiểu 1: Hiển thị bounding box và tên người trên webcam
    Chạy lệnh: $ python recognition.py
  - Kiểu 2:  Hiển thị bounding box và tên người trên video:
    + Đưa video vào folder raw
    + Chạy lệnh: $ python video_recognition.py
    + Nhập tên video
