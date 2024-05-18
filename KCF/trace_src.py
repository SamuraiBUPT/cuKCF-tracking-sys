import sys
import trace


def run_script(script_name):
    tracer = trace.Trace(
        ignoredirs=[sys.prefix, sys.exec_prefix],
        trace=1,
        count=0,
    )

    # 打开文件以进行写入，并设置编码为utf-8
    with open('trace_output.txt', 'w', encoding='utf-8') as f:
        sys.stdout = f  # Redirect stdout to file

        # 确保脚本文件存在并可以读取
        try:
            with open(script_name, 'r', encoding='utf-8') as script_file:
                script_code = script_file.read()
        except FileNotFoundError:
            print(f"Error: The file {script_name} was not found.")
            return
        except Exception as e:
            print(
                f"Error: An error occurred while reading the file {script_name}: {e}")
            return

        # 运行并跟踪脚本代码
        try:
            tracer.run(script_code)
        except Exception as e:
            print(f"Error: An error occurred while running the script: {e}")

        sys.stdout = sys.__stdout__  # Reset redirect


if __name__ == '__main__':
    script_name = 'kcf_v4.py'  # 替换为你的脚本名
    run_script(script_name)
