from setuptools import find_packages, setup

setup(
    name="spot_tools",
    version="0.0.1",
    url="",
    author="",
    author_email="",
    description="Tools for Spot",
    package_dir={"": "src"},
    packages=find_packages("src"),
    package_data={"": ["*.yaml", "*.jpg"]},
    install_requires=[
        "numpy",
        "matplotlib",
        "pyyaml",
        "bosdyn-client",
        "scikit-image",
        # opencv-python, not opencv-python-headless: both provide cv2, and the
        # perception stack this runs beside pulls opencv-python via ultralytics.
        # Declaring the headless build made a normal install replace the other
        # one, so whichever landed last decided which cv2 everything got.
        "opencv-python",
        "shapely",
        "onnxruntime",
        "transforms3d",
        # Reached through spot_executor.spot -> spot_executor.stitch_front_images.
        # Previously undeclared, so the executor node died at import with a
        # ModuleNotFoundError on a machine that had not installed them by hand.
        "pygame",
        "PyOpenGL",
        "Pillow",
    ],
    extras_require={
        # Development-only; not needed to run the executor on a robot.
        "dev": ["pytest", "pre-commit"],
    },
)
