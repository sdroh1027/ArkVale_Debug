import torch

# 1. (2, 3) 형태의 2차원 텐서 생성 (float32, 4 bytes)
tensor = torch.arange(6, dtype=torch.float32).reshape(2, 3)

print(f"Tensor shape: {tensor.shape}")
print(f"Tensor data:\n{tensor}\n")

# 2. 각 요소의 가상 메모리 주소(VA) 확인
base_addr = tensor.data_ptr()
print(f"Base Address (Start): {base_addr}")

print("-" * 50)
print(f"{'Index':<10} | {'Value':<5} | {'Address (VA)':<20} | {'Offset from Start':<10}")
print("-" * 50)

# 평탄화(flatten)된 순서대로 메모리 주소가 4씩 증가하는지 확인
flat_data = tensor.view(-1)
for i in range(tensor.numel()):
    element_addr = flat_data[i].data_ptr()
    offset = element_addr - base_addr
    print(f"Flat[{i}]    | {flat_data[i].item():<5.0f} | {element_addr:<20} | {offset} bytes")



