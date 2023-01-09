from pdf2image import convert_from_path
img = 'choice_distribution.pdf'
images = convert_from_path(img, 500)
for image in images:
    image.save(img.replace('pdf', 'png'))
