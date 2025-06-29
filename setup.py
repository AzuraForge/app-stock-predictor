from setuptools import setup, find_packages
setup(
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    # DÜZELTME: Paket kurulduğunda .yml gibi Python dışı dosyaların da
    # kopyalanmasını sağlar. Bu, API ve Worker loglarındaki hatayı çözer.
    include_package_data=True, 
    package_data={
        # "azuraforge_stockapp" paketi içindeki tüm .yml dosyalarını dahil et.
        "azuraforge_stockapp": ["config/*.yml"], 
    },
)