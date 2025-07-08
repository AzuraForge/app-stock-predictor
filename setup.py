from setuptools import setup, find_packages

setup(
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    
    # Bu ayar, MANIFEST.in dosyasının okunmasını sağlar.
    # Bu, veri dosyalarını dahil etmenin en güvenilir yoludur.
    include_package_data=True,
)