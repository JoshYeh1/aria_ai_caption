examples = [
    "Describe this image.",
    "What can you see in this picture?",
    "How many objects are there?",
    "What is the person doing?",
    "Describe the scenery in detail.",
    "Explain what happens in this chart.",
    "Summarize the content of the photo.",
    "What colors dominate the image?",
    "What emotions does the image convey?",
    "Write a caption for this visual."
]

with open("calib.txt", "w") as f:
    for line in examples:
        f.write(line + "\n")
