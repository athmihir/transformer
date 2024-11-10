import tiktoken
import time
import torch 

start_time = time.time()
enc = tiktoken.encoding_for_model("gpt-4o")

with open('data/books_large_p1.txt', 'r', encoding='utf-8') as file:
    output_tensors = []
    chunk_count = 0
    while True:
        chunk = file.read(1024*1024) # Read 1MB of data at a time.
        if not chunk:
            break
        encoded_text = enc.encode(chunk)
        output_tensors.append(torch.tensor(encoded_text, dtype=torch.long))
        chunk_count += 1
        print(f"Chunk number {chunk_count} completed")
    
    # Concatenating all the tensors
    print("concatenating all the tensors...")
    final_tensor = torch.cat(output_tensors)
    # Save the final tensor
    print("saving the final tensor...")
    torch.save(final_tensor, "data/input_tensor1.pt")

# Timer
end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")