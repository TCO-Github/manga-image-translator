import asyncio
import argparse
import time
from PIL import Image
import cv2
import numpy as np
import requests
import os
from oscrypto import util as crypto_utils
import asyncio
import torch

from detection import dispatch as dispatch_detection, load_model as load_detection_model
from ocr import OCRS, dispatch as dispatch_ocr, prepare as prepare_ocr
from inpainting import INPAINTERS, dispatch as dispatch_inpainting, prepare as prepare_inpainting
from translators import OFFLINE_TRANSLATORS, TRANSLATORS, VALID_LANGUAGES, dispatch as dispatch_translation, prepare as prepare_translation
from text_mask import dispatch as dispatch_mask_refinement
from textline_merge import dispatch as dispatch_textline_merge
from upscaling import dispatch as dispatch_upscaling, prepare as prepare_upscaling
from text_rendering import dispatch as dispatch_rendering, text_render
from textblockdetector import dispatch as dispatch_ctd_detection, load_model as load_ctd_model
from textblockdetector.textblock import visualize_textblocks
from utils import load_image, dump_image

parser = argparse.ArgumentParser(description='Seamlessly translate mangas into a chosen language')
parser.add_argument('-m', '--mode', default='demo', type=str, choices=['demo', 'batch', 'web', 'web2'], help='Run demo in either single image demo mode (demo), web service mode (web) or batch translation mode (batch)')
parser.add_argument('-i', '--image', default='', type=str, help='Path to an image file if using demo mode, or path to an image folder if using batch mode')
parser.add_argument('-o', '--image-dst', default='', type=str, help='Path to the destination folder for translated images in batch mode')
parser.add_argument('-l', '--target-lang', default='ENG', type=str, choices=VALID_LANGUAGES, help='Destination language')
parser.add_argument('-v', '--verbose', action='store_true', help='Print debug info and save intermediate images')
parser.add_argument('--host', default='127.0.0.1', type=str, help='Used by web module to decide which host to attach to')
parser.add_argument('--port', default=5003, type=int, help='Used by web module to decide which port to attach to')
parser.add_argument('--log-web', action='store_true', help='Used by web module to decide if web logs should be surfaced')
parser.add_argument('--ocr', default='48px_ctc', type=str, choices=OCRS, help='Optical character recognition (OCR) model to use')
parser.add_argument('--inpainter', default='lama_mpe', type=str, choices=INPAINTERS, help='Inpainting model to use')
parser.add_argument('--translator', default='google', type=str, choices=TRANSLATORS, help='Language translator to use')
parser.add_argument('--use-cuda', action='store_true', help='Turn on/off cuda')
parser.add_argument('--use-cuda-limited', action='store_true', help='Turn on/off cuda (excluding offline translator)')
parser.add_argument('--detection-size', default=1536, type=int, help='Size of image used for detection')
parser.add_argument('--inpainting-size', default=2048, type=int, help='Size of image used for inpainting (too large will result in OOM)')
parser.add_argument('--unclip-ratio', default=2.3, type=float, help='How much to extend text skeleton to form bounding box')
parser.add_argument('--box-threshold', default=0.7, type=float, help='Threshold for bbox generation')
parser.add_argument('--text-threshold', default=0.5, type=float, help='Threshold for text detection')
parser.add_argument('--text-mag-ratio', default=10, type=int, help='Text rendering magnification ratio, larger means higher quality')
parser.add_argument('--font-size-offset', default=0, type=int, help='Offset font size by a given amount, positive number increase font size and vice versa')
parser.add_argument('--force-horizontal', action='store_true', help='Force text to be rendered horizontally')
parser.add_argument('--force-vertical', action='store_true', help='Force text to be rendered vertically')
parser.add_argument('--upscale-ratio', default=None, type=int, choices=[1, 2, 4, 8, 16, 32], help='waifu2x image upscale ratio')
parser.add_argument('--use-ctd', action='store_true', help='Use comic-text-detector for text detection')
parser.add_argument('--manga2eng', action='store_true', help='Render english text translated from manga with some typesetting')
parser.add_argument('--eng-font', default='fonts/comic shanns 2.ttf', type=str, help='Path to font used by manga2eng mode')
args = parser.parse_args()

def update_state(task_id, nonce, state):
	while True:
		try:
			requests.post(f'http://{args.host}:{args.port}/task-update-internal', json = {'task_id': task_id, 'nonce': nonce, 'state': state}, timeout = 20)
			return
		except Exception:
			if 'error' in state or 'finished' in state:
				continue
			else:
				break

def get_task(nonce):
	try:
		rjson = requests.get(f'http://{args.host}:{args.port}/task-internal?nonce={nonce}', timeout = 3600).json()
		if 'task_id' in rjson and 'data' in rjson:
			return rjson['task_id'], rjson['data']
		elif 'data' in rjson:
			return None, rjson['data']
		return None, None
	except Exception:
		return None, None

async def infer(
	image: Image.Image,
	mode,
	nonce = '',
	options = None,
	task_id = '',
	dst_image_name = '',
	):

	img, alpha_ch = load_image(image)

	options = options or {}
	img_detect_size = args.detection_size

	if 'size' in options:
		size_ind = options['size']
		if size_ind == 'S':
			img_detect_size = 1024
		elif size_ind == 'M':
			img_detect_size = 1536
		elif size_ind == 'L':
			img_detect_size = 2048
		elif size_ind == 'X':
			img_detect_size = 2560

	if 'detector' in options:
		detector = options['detector']
	else:
		detector = 'ctd' if args.use_ctd else 'default'

	render_text_direction_overwrite = options.get('direction')
	if not render_text_direction_overwrite:
		if args.force_horizontal:
			render_text_direction_overwrite = 'h'
		elif args.force_vertical:
			render_text_direction_overwrite = 'v'
		else:
			render_text_direction_overwrite = 'auto'

	src_lang = 'auto'
	if 'tgt' in options:
		tgt_lang = options['tgt']
	else:
		tgt_lang = args.target_lang
	if 'translator' in options:
		translator = options['translator']
	else:
		translator = args.translator
	
	if not dst_image_name:
		dst_image_name = f'result/{task_id}/final.png'

	print(f' -- Detection resolution {img_detect_size}')
	print(f' -- Detector using {detector}')
	print(f' -- Render text direction is {render_text_direction_overwrite}')

	print(' -- Preparing translator')
	await prepare_translation(translator, src_lang, tgt_lang)

	#0 - Upscaling the Image
	print(' -- Preparing upscaling') 
	await prepare_upscaling('waifu2x', args.upscale_ratio)

	if args.upscale_ratio or image.size[0] < 800 or image.size[1] < 800:
		print(' -- Running upscaling')
		if mode == 'web' and task_id:
			update_state(task_id, nonce, 'upscaling')

		if args.upscale_ratio:
			img_upscaled_pil = (await dispatch_upscaling('waifu2x', [image], args.upscale_ratio, args.use_cuda))[0]
			img, alpha_ch = load_image(img_upscaled_pil)
		elif image.size[0] < 800 or image.size[1] < 800:
			ratio = max(4, 800 / image.size[0], 800 / image.size[1])
			img_upscaled_pil = (await dispatch_upscaling('waifu2x', [image], ratio, args.use_cuda))[0]
			img, alpha_ch = load_image(img_upscaled_pil)

	print(' -- Running text detection')
	if mode == 'web' and task_id:
		update_state(task_id, nonce, 'detection')

	#1 - Text and text Detection
	if detector == 'ctd':
		mask, final_mask, textlines = await dispatch_ctd_detection(img, args.use_cuda)
	else:
		#Generates a Raw 'mask' output, which contains all the detected text on the image, and generates detected Textlines output which contains all the textlines to be filtered
		textlines, mask = await dispatch_detection(img, img_detect_size, args.use_cuda, args, verbose = args.verbose)

	#Visualize the unfiltered textlines with Red, and also save a RAW Mask of unfiltered detection (Raw Mask will contain almost all of the text on an image, but may have errors)
	if args.verbose:
		if detector == 'ctd':
			bboxes = visualize_textblocks(cv2.cvtColor(img,cv2.COLOR_BGR2RGB), textlines)
			cv2.imwrite(f'result/{task_id}/bboxes.png', bboxes)
			cv2.imwrite(f'result/{task_id}/mask_raw.png', mask)
		else:
			img_bbox_raw = np.copy(img)
			for txtln in textlines:
				cv2.polylines(img_bbox_raw, [txtln.pts], True, color = (255, 0, 0), thickness = 2)
			print('Saving bboxes_unfiltered to: result/{task_id}/bboxes_unfiltered.png')
			cv2.imwrite(f'result/{task_id}/bboxes_unfiltered.png', cv2.cvtColor(img_bbox_raw, cv2.COLOR_RGB2BGR))
			print('Saving Mask to: result/{task_id}/mask_raw.png')
			cv2.imwrite(f'result/{task_id}/mask_raw.png', mask)

	print('Text Detection Complete')
	return textlines, mask


async def infer_safe(
	img: Image.Image,
	mode,
	nonce,
	options = None,
	task_id = '',
	dst_image_name = '',
	):
	try:
		return await infer(
			img,
			mode,
			nonce,
			options,
			task_id,
			dst_image_name,
		)
	except Exception:
		import traceback
		traceback.print_exc()
		update_state(task_id, nonce, 'error')

def replace_prefix(s: str, old: str, new: str):
	if s.startswith(old):
		s = new + s[len(old):]
	return s

async def main(mode = 'demo'):
	print(' -- Preload Checks')
	if args.use_cuda_limited:
		args.use_cuda = True
	if not torch.cuda.is_available() and args.use_cuda:
		raise Exception('CUDA compatible device could not be found while %s args was set...'
						% ('--use_cuda_limited' if args.use_cuda_limited else '--use_cuda'))

	print(' -- Loading models')
	os.makedirs('result', exist_ok=True)
	text_render.prepare_renderer()
	await prepare_ocr(args.ocr, args.use_cuda)
	load_ctd_model(args.use_cuda)
	load_detection_model(args.use_cuda)
	await prepare_inpainting(args.inpainter, args.use_cuda)

	print(' -- Running in single image Text Detection')
	if not args.image:
		print('please provide an image')
		parser.print_usage()
		return
	return await infer(Image.open(args.image), mode)

if __name__ == '__main__':
	try:
		print(args)
		loop = asyncio.new_event_loop()
		asyncio.set_event_loop(loop)
		loop.run_until_complete(main(args.mode))
	except KeyboardInterrupt:
		print()
