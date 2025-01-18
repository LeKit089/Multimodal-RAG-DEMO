from .cpm_convertor import CPMConvertorFactory

def convert_img_to_tables(image_path):
    """
    Convert an image to table data using the MiniCPM model.
    """
    # Get the singleton instance of CPMConvertor
    factory = CPMConvertorFactory()
    cpm_convertor = factory.get_instance()
    
    # Define the query for the model
    question = "请你将图像转换为markdown形式的表格"
    
    # Process the image
    converted_text = cpm_convertor.convert(query=question, img_path=image_path)
    
    # Clear GPU memory
    cpm_convertor.clear_GPU_mem()
    
    return converted_text