from paddleocr import PaddleOCRVL

# 英伟达 GPU
pipeline = PaddleOCRVL()
output = pipeline.predict(rf"C:\Users\11\Desktop\pj\image_extract\imgs_test\resist.png")
print(f"*******{output[0]}*********", type(output))
# 格式化输出output，以及内部的字典
print(output[0].get("parsing_res_list")[0], type(output[0].get("parsing_res_list")[0]))
for res in output:
    res.print() ## 打印预测的结构化输出