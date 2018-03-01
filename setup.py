from setuptools import setup

setup(name='danzem',
      version='0.3.3',
      description='Data Analysis tools for the New Zealand Electricity Market',
      url='https://github.com/AlanAnsell/DANZEM',
      author='Alan Ansell',
      install_requires=[
          'numpy',
          'pandas',
          'pytz'
      ],
      test_suite='nose.collector',
      tests_require=['nose'],
      packages=['danzem'],
      include_package_data=True,
      zip_safe=False)

