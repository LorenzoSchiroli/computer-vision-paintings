import project
import argparse
import sys

def arg_parse():
	parser = argparse.ArgumentParser(description='Computer Vision Project')

	parser.add_argument("--video", dest='video', help="If false, it takes an image in input", default="False")
	parser.add_argument("--input", dest="inputPath", help="Input video path", default="")
	parser.add_argument("--output", dest="outputPath", help="Output video path without .avi extension", default="")
	parser.add_argument("--showrect", dest="showRect", help="If true, it shows the rectification in output", default="False")
	return parser.parse_args()


if __name__ == '__main__':
	args = arg_parse()
	video = str(args.video)
	inputPath = str(args.inputPath)
	if inputPath == "":
		print("The input file is empty")
		sys.exit()

	showRect = str(args.showRect)
	if showRect == "False":
		showRect = False
	else:
		showRect = True

	if video == "True":
		outputPath = str(args.outputPath)
		if outputPath == "":
			print("The output video file is empty")
			sys.exit()
		if outputPath.endswith(".avi"):
			print("The output video file must not have the .avi extension")
			sys.exit()
		project.executeVideo(inputPath, outputPath, showRect)
	else:
		project.executeImage(inputPath, showRect)
	print("End")