from functools import wraps
import os
import requests
import shutil
import time
import traceback
import tqdm


READONLY_DIR = os.path.expanduser('~/src/DeepLearning/dltemplate/readonly/')

tqdm.monitor_interval = 0  # workaround for https://github.com/tqdm/tqdm/issues/481


# https://www.saltycrane.com/blog/2009/11/trying-out-retry-decorator-python/
def retry(exception_to_check, tries=4, delay=3, backoff=2):
    def deco_retry(f):

        @wraps(f)
        def f_retry(*args, **kwargs):
            tries_, delay_ = tries, delay
            while tries_ > 1:
                try:
                    return f(*args, **kwargs)
                except KeyboardInterrupt as e:
                    raise e
                except exception_to_check as e:
                    print('%s, retrying in %d seconds...' % (str(e), delay_))
                    traceback.print_exc()
                    time.sleep(delay_)
                    tries_ -= 1
                    delay_ *= backoff
            return f(*args, **kwargs)

        return f_retry  # true decorator

    return deco_retry


@retry(Exception)
def download_file(url, file_path):
    if not os.path.exists(file_path):
        r = requests.get(url, stream=True)
        total_size = int(r.headers.get('content-length') or 0)  # in case content-length isn't set
        bar = tqdm.tqdm_notebook(total=total_size, unit='B', unit_scale=True)
        bar.set_description(os.path.split(file_path)[-1])
        incomplete_download = False
        try:
            with open(file_path, 'wb', buffering=16 * 1024 * 1024) as f:
                for chunk in r.iter_content(1 * 1024 * 1024):
                    f.write(chunk)
                    bar.update(len(chunk))
        except Exception as e:
            raise e
        finally:
            bar.close()
            if os.path.exists(file_path) and 0 < total_size != os.path.getsize(file_path):
                incomplete_download = True
                os.remove(file_path)

        if incomplete_download:
            raise Exception('Incomplete download')
    else:
        print(file_path, 'already exists')


def download_from_github(version, filename, target_dir):
    url = 'https://github.com/hse-aml/intro-to-dl/releases/download/{0}/{1}'.format(version, filename)
    file_path = os.path.join(target_dir, filename)
    download_file(url, file_path)


def sequential_downloader(version, filenames, target_dir):
    os.makedirs(target_dir, exist_ok=True)
    for filename in filenames:
        download_from_github(version, filename, target_dir)


def link_all_files_from_dir(src_dir, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    for fn in os.listdir(src_dir):
        src_file = os.path.join(src_dir, fn)
        dst_file = os.path.join(dst_dir, fn)
        if os.name == 'nt':
            shutil.copyfile(src_file, dst_file)
        else:
            if os.path.islink(dst_file):
                os.remove(dst_file)

            os.symlink(os.path.abspath(src_file), dst_file)


def link_all_keras_resources():
    link_all_files_from_dir(os.path.expanduser('~/.keras/datasets'), READONLY_DIR + 'keras/datasets/')
    link_all_files_from_dir(os.path.expanduser('~/.keras/models'), READONLY_DIR + 'keras/models/')
