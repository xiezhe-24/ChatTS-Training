from llamafactory.chat import ChatModel
from llamafactory.extras.misc import torch_gc
import json
import re


def extract_and_remove_ts(s):
    pattern = r'(<ts>)(.*?)(<ts/>)'
    matches = re.findall(pattern, s)
    extracted_lists = [json.loads(match[1]) for match in matches]
    modified_s = re.sub(pattern, r'\1\3', s)
    
    if len(extracted_lists) == 0:
        extracted_lists = None
    return modified_s, extracted_lists


try:
    import platform

    if platform.system() != "Windows":
        import readline  # noqa: F401
except ImportError:
    print("Install `readline` for a better experience.")


def main():
    chat_model = ChatModel()
    messages = []
    timeseries = None
    extract_ts_flag = True
    print("Welcome to the CLI application, use `clear` to remove the history, use `exit` to exit the application.")

    while True:
        try:
            query = input("\nUser: ")
            query = query.replace('\\n', '\n')

            if query.strip() == "nots":
                extract_ts_flag = False
                print("extract_ts_flag has been set to False.")
                continue

            if extract_ts_flag:
                query, cur_ts = extract_and_remove_ts(query)
            else:
                cur_ts = None
            if cur_ts is not None and len(cur_ts) > 0:
                if timeseries is None:
                    timeseries = []
                timeseries.extend(cur_ts)
        except UnicodeDecodeError:
            print("Detected decoding error at the inputs, please set the terminal encoding to utf-8.")
            continue
        except Exception:
            raise

        if query.strip() == "exit":
            break

        if query.strip() == "clear":
            messages = []
            timeseries = None
            torch_gc()
            print("History has been removed.")
            continue

        print(f"[DEBUG] timeseries: {len(timeseries) if timeseries is not None else None}")
        messages.append({"role": "user", "content": query})
        print("Assistant: ", end="", flush=True)
        print("----------------------------------------", flush=True)

        response = ""
        for new_text in chat_model.stream_chat(messages, timeseries=timeseries):
            print(new_text, end="", flush=True)
            response += new_text
        print()
        messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
