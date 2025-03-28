from models.car2d import Robot2D
from source.abstraction import Abstraction

prism_executable='./prism-4.8.1-mac64-arm/bin/prism'
foldername='./output'

abstraction = Abstraction(Robot2D(), './models/car2d.conf')

abstraction.generate_noisy_observations()
abstraction.find_actions()
abstraction.find_transitions()
abstraction.create_IMDP(foldername=foldername)
abstraction.solve_iMDP(foldername=foldername, prism_executable=prism_executable)