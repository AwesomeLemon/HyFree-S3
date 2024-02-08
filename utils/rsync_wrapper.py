import socket
import subprocess
import time
from pathlib import Path


class RsyncWrapper:
    '''
    Uses rsync to upload/download files. Assumes established ssh connection, with agent running (so, no password)
    Note that gethostname() returns full name ('first.second.third', not 'first')
    '''

    def __init__(self, ssh_user, ray_head_node, if_shared_fs, final_upload_node):
        self.ssh_user = ssh_user
        self.ray_head_node = ray_head_node
        self.if_shared_fs = if_shared_fs
        self.final_upload_node = final_upload_node

    def _get_name_this_node(self):
        return socket.gethostname()

    def download(self, path, if_dir=False):
        if self.if_shared_fs:
            return
        if self.ray_head_node == self._get_name_this_node():
            return

        try:
            # rsync has trouble with creating intermediate dirs
            parent_directories = list(Path(path).parents)
            for parent_dir in reversed(parent_directories):
                if not parent_dir.exists():  # better to check before calling mkdir to not run into permission troubles
                    parent_dir.mkdir(exist_ok=True)

            output_path = path
            if if_dir:
                output_path = Path(path).parent.absolute()

            command = ['rsync', '-hzavP', f'{self.ssh_user}@{self.ray_head_node}:{path}', output_path]
            result = subprocess.run(command)
            while result.returncode != 0:
                print('rsync failed, waiting for 5s and retrying')
                print(f'{self._get_name_this_node()=} {command=}')
                time.sleep(5)
                result = subprocess.run(command)
        except Exception as e:  # it may be alright that the file doesn't exist
            print(e)

    def upload(self, path, if_dir=False, if_delete=False):
        if self.if_shared_fs:
            return
        if self.ray_head_node == self._get_name_this_node():
            return
        # print(f'rsync: uploading {path}')

        try:
            # rsync has trouble with creating intermediate dirs
            parent_directories = list(Path(path).parents)
            for parent_dir in reversed(parent_directories):
                if not parent_dir.exists():  # better to check before calling mkdir to not run into permission troubles
                    parent_dir.mkdir(exist_ok=True)

            output_path = path
            if if_dir:
                output_path = Path(path).parent.absolute()
            command = ['rsync', '-hzaqP']
            if if_delete:
                command += ['--delete']
            command += [path, f'{self.ssh_user}@{self.ray_head_node}:{output_path}']
            result = subprocess.run(command)
            while result.returncode != 0:
                print('rsync failed, waiting for 5s and retrying')
                print(f'{self._get_name_this_node()=} {command=}')
                time.sleep(5)
                result = subprocess.run(command)
        except Exception as e:
            print(e)

    def upload_final(self, path, if_dir=False):
        if self.final_upload_node == self._get_name_this_node():
            return

        try:
            # rsync has trouble with creating intermediate dirs
            parent_directories = list(Path(path).parents)
            for parent_dir in reversed(parent_directories):
                if not parent_dir.exists():  # better to check before calling mkdir to not run into permission troubles
                    parent_dir.mkdir(exist_ok=True)
            output_path = path
            if if_dir:
                output_path = Path(path).parent.absolute()
            subprocess.run(['rsync', '-hzaqP',
                            path, f'{self.ssh_user}@{self.final_upload_node}:{output_path}'])
        except Exception as e:
            print(e)