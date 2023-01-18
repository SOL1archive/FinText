def divide_chunks(lt, n):
     
    for i in range(0, len(lt), n):
        yield lt[i:i + n]

def to(tensor, device):
    tensor.to(device)
    