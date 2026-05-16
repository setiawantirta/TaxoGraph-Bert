from setuptools import setup

setup(
    name='pyTirs',
    version='0.0.1',    
    description='Package Python perkuliahan yang diampu Tirta Setiawan di Prodi Sains Data Fakultas Sains Institut Teknologi Sumatera',
    url='https://github.com/setiawantirta/pyTirs',
    author='tirta setiawan',
    author_email='tirta.setiawan@sd.itera.ac.id',    
    license='BSD 2-clause',
    packages=['pyTirs'],
    install_requires=['click', 
                      'pytz'
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',  
        'Operating System :: OS Independent',    
        'Programming Language :: Python :: 3',
    ],
    entry_points={"console_scripts": ["cloudquicklabs1 = src.main:main"]},
)  
