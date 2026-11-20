from Text_exreact import Transcriber
from Make_note import make_notes
import os
import av


def main():
    usable_files = [file for file in os.listdir("./input/") if file.endswith(tuple(av.formats_available))]
    if not usable_files:
        raise FileNotFoundError("No audio file available under input dir")
    print("Select a file-")
    for i in range(len(usable_files)):
        print(f"{i}-{usable_files[i]}")
    file_no = int(input("Enter the value: "))
    context = input("Context[if no context leave empty]: ")
    extractor = Transcriber("medium")
    text = extractor.transcribe(os.path.join("./input/", usable_files[file_no]))
    usable_files = [os.path.splitext(file)[0] for file in usable_files]
    with open(f"./output/{usable_files[file_no]}.txt", 'w', encoding="utf-8") as f:
        f.write(text)
    note = make_notes(text, context)
    with open(f"./output/{usable_files[file_no]}.md", 'w', encoding="utf-8") as f:
        f.write(note)
    
main()