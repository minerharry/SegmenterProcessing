from zipfile import ZipFile,ZIP_DEFLATED
from pathlib import Path

with ZipFile("d2s.zip") as archive:
    archive.printdir();

def _python_cmd_zip(source,dest,recurse=False,compresslevel=6):
    source = Path(source);
    dest = Path(dest)
    with ZipFile(dest,'w',compression=ZIP_DEFLATED,compresslevel=compresslevel) as archive:
        for filepath in (source.rglob("*") if recurse else source.iterdir()):
            archive.write(filepath,arcname=filepath.relative_to(source));

def _python_cmd_unzip(source,dest,overwrite=False):
    source = Path(source);
    dest = Path(dest);
    with ZipFile(source,'r') as archive:
        for member in archive.infolist():
            file_path = dest/member.filename
            if not file_path.exists():
                archive.extract(member, dest)

# def add(archive,path:Path):
#     for filepath in path.iterdir():
#         archive.write(filepath,arcname=filepath.name);

# def add_recursive(archive,path:Path):
#     for file_path in path.rglob("*"):
#         archive.write(
#             file_path,
#             arcname=file_path.relative_to(path)
#         )
