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

from detection.ctd_utils import TextBlock
from typing import List, Union
from detection import DETECTORS, dispatch as dispatch_detection, prepare as prepare_detection
from detection.ctd_utils.textblock import visualize_textblocks
from ocr import OCRS, dispatch as dispatch_ocr, prepare as prepare_ocr
from inpainting import INPAINTERS, dispatch as dispatch_inpainting, prepare as prepare_inpainting
from translators import OFFLINE_TRANSLATORS, TRANSLATORS, VALID_LANGUAGES, dispatch as dispatch_translation, prepare as prepare_translation
from upscaling import dispatch as dispatch_upscaling, prepare as prepare_upscaling
from textmask_refinement import dispatch as dispatch_mask_refinement
from text_rendering import text_render, dispatch as dispatch_rendering
from text_rendering.text_render import count_valuable_text
from utils import load_image, dump_image

parser = argparse.ArgumentParser(description='Seamlessly translate mangas into a chosen language')
parser.add_argument('-m', '--mode', default='demo', type=str, choices=['demo', 'batch', 'web', 'web2', 'ws'], help='Run demo in either single image demo mode (demo), web service mode (web) or batch translation mode (batch)')
parser.add_argument('-i', '--image', default='', type=str, help='Path to an image file if using demo mode, or path to an image folder if using batch mode')
parser.add_argument('-o', '--image-dst', default='', type=str, help='Path to the destination folder for translated images in batch mode')
parser.add_argument('-l', '--target-lang', default='CHS', type=str, choices=VALID_LANGUAGES, help='Destination language')
parser.add_argument('-v', '--verbose', action='store_true', help='Print debug info and save intermediate images')
parser.add_argument('--host', default='127.0.0.1', type=str, help='Used by web module to decide which host to attach to')
parser.add_argument('--port', default=5003, type=int, help='Used by web module to decide which port to attach to')
parser.add_argument('--log-web', action='store_true', help='Used by web module to decide if web logs should be surfaced')
parser.add_argument('--detector', default='default', type=str, choices=DETECTORS, help='Text detector used for creating a text mask from an image')
parser.add_argument('--ocr', default='48px_ctc', type=str, choices=OCRS, help='Optical character recognition (OCR) model to use')
parser.add_argument('--inpainter', default='lama_mpe', type=str, choices=INPAINTERS, help='Inpainting model to use')
parser.add_argument('--translator', default='google', type=str, choices=TRANSLATORS, help='Language translator to use')
parser.add_argument('--mtpe', action='store_true', help='Turn on/off machine translation post editing (MTPE) on the command line (works only on linux right now)')
parser.add_argument('--use-cuda', action='store_true', help='Turn on/off cuda')
parser.add_argument('--use-cuda-limited', action='store_true', help='Turn on/off cuda (excluding offline translator)')
parser.add_argument('--detection-size', default=1536, type=int, help='Size of image used for detection')
parser.add_argument('--det-rearrange-max-batches', default=4, type=int, help='Max batch size produced by the rearrangement of image with extreme aspectio, reduce it if cuda OOM')
parser.add_argument('--inpainting-size', default=2048, type=int, help='Size of image used for inpainting (too large will result in OOM)')
parser.add_argument('--unclip-ratio', default=2.3, type=float, help='How much to extend text skeleton to form bounding box')
parser.add_argument('--box-threshold', default=0.7, type=float, help='Threshold for bbox generation')
parser.add_argument('--text-threshold', default=0.5, type=float, help='Threshold for text detection')
parser.add_argument('--text-mag-ratio', default=1, type=int, help='Text rendering magnification ratio, larger means higher quality')
parser.add_argument('--font-size-offset', default=0, type=int, help='Offset font size by a given amount, positive number increase font size and vice versa')
parser.add_argument('--force-horizontal', action='store_true', help='Force text to be rendered horizontally')
parser.add_argument('--force-vertical', action='store_true', help='Force text to be rendered vertically')
parser.add_argument('--upscale-ratio', default=None, type=int, choices=[1, 2, 4, 8, 16, 32], help='waifu2x image upscale ratio')
parser.add_argument('--manga2eng', action='store_true', help='Render english text translated from manga with some typesetting')
parser.add_argument('--ws-url', default='ws://localhost:5000', type=str, help='Server URL for WebSocket mode')
parser.add_argument('--font-path', default='', type=str, help='Path to font file, subsequent fonts (seperated by commas) will be used as backup')
args = parser.parse_args()

async def noop(*args, **kwargs):
  pass

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
    update_state = noop,
):

    img_rgb, img_alpha = load_image(image)

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
        detector = args.detector

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

    if 'boxvalue' in options:
        boxvalue = float(options['boxvalue'])
        print(f'Box Value Specified as {boxvalue}')
        args.box_threshold = boxvalue

    if 'textvalue' in options:
        textvalue = float(options['textvalue'])
        print(f'Text Value Specified as {textvalue}')
        args.text_threshold = textvalue

    print(f' -- Detection resolution {img_detect_size}')
    print(f' -- Detector using {detector}')
    print(f' -- Render text direction is {render_text_direction_overwrite}')

    print(' -- Preparing translator')
    await prepare_translation(translator, src_lang, tgt_lang)

    # The default text detector doesn't work very well on smaller images, so small images
    # will get upscaled automatically unless --upscale-ratio=1 was set
    if args.upscale_ratio:
        print(' -- Running upscaling')
        await update_state('upscaling')
        img_upscaled_pil = (await dispatch_upscaling('waifu2x', [image], args.upscale_ratio, args.use_cuda))[0]
        img_rgb, img_alpha = load_image(img_upscaled_pil)

    print(' -- Running text detection')
    await update_state('detection')
    text_regions, mask = await dispatch_detection(detector, img_rgb, img_detect_size, args.text_threshold, args.box_threshold,
                                                  args.unclip_ratio, args.det_rearrange_max_batches, args.use_cuda, args.verbose)
    if not text_regions:
        print('No text regions were detected - Skipping')
        image.save(dst_image_name)
        return
    if args.verbose:
        bboxes = visualize_textblocks(cv2.cvtColor(img_rgb,cv2.COLOR_BGR2RGB), text_regions)
        cv2.imwrite(f'result/{task_id}/bboxes.png', bboxes)

    print(' -- Running OCR')
    await update_state('ocr')
    text_regions = await dispatch_ocr(args.ocr, img_rgb, text_regions, args.use_cuda, args.verbose)

    # filter regions by their text
    text_regions = list(filter(lambda r: count_valuable_text(r.get_text()) > 1 and not r.get_text().isnumeric(), text_regions))
    if detector == 'default':
        await update_state('mask_generation')
        cv2.imwrite(f'result/{task_id}/mask_raw.png', mask)
        mask = await dispatch_mask_refinement(text_regions, img_rgb, mask)

    # in web mode, we can start online translation tasks async
    if mode == 'web' and task_id and options.get('translator') not in OFFLINE_TRANSLATORS:
        await update_state('translating')
        requests.post(f'http://{args.host}:{args.port}/request-translation-internal', json = {'task_id': task_id, 'nonce': nonce, 'texts': [{'s': f"{str(i+1)}: {r.get_text()}", 'f': str(r.font_size), "x": r.xyxy[0], "y": r.xyxy[1]} for i, r in enumerate(text_regions)]}, timeout = 20)

    print(' -- Running inpainting')
    await update_state('inpainting')
    img_inpainted = await dispatch_inpainting(args.inpainter, img_rgb, mask, args.inpainting_size, args.use_cuda, args.verbose)

    if args.verbose:
        cv2.imwrite(f'result/{task_id}/mask_final.png', mask)
        cv2.imwrite(f'result/{task_id}/inpainted.png', cv2.cvtColor(img_inpainted, cv2.COLOR_RGB2BGR))

    print(' -- Translating')
    translated_sentences = None
    if mode != 'web' or translator in OFFLINE_TRANSLATORS:
        await update_state('translating')
        queries = [r.get_text() for r in text_regions]
        translated_sentences = await dispatch_translation(translator, src_lang, tgt_lang, queries, args.mtpe, use_cuda=args.use_cuda and not args.use_cuda_limited)
    else:
        # wait for at most 1 hour for manual translation
        if options.get('manual', False):
            wait_for = 3600
        else:
            wait_for = 30 # 30 seconds for machine translation
        wait_until = time.time() + wait_for
        while time.time() < wait_until:
            ret = requests.post(f'http://{args.host}:{args.port}/get-translation-result-internal', json = {'task_id': task_id, 'nonce': nonce}, timeout = 20).json()
            if 'result' in ret:
                translated_sentences = ret['result']
                if isinstance(translated_sentences, str):
                    if translated_sentences == 'error':
                        await update_state('error-lang')
                        return
                break
            if 'result_preview' in ret:
                print(ret['result_preview'])
                translated_sentences = ret['result_preview']
                output = await GenerateRender(translated_sentences, mode, task_id, nonce, tgt_lang, img_inpainted, img_rgb, text_regions, render_text_direction_overwrite)

                print(' -- Saving results')
                img_pil = dump_image(output, img_alpha)
                img_pil.save(f'result/{task_id}/final_preview.png')

                if mode == 'web' and task_id:
                    update_state(task_id, nonce, 'finished-preview')

                if isinstance(translated_sentences, str):
                    if translated_sentences == 'error':
                        update_state(task_id, nonce, 'error-lang')
                        return
                await asyncio.sleep(2)

    print(' -- Rendering translated text')
    if translated_sentences == None:
        print("No text found!")
        await update_state('error-no-txt')
        return

    # render translated texts
    await update_state('render')

    print(' -- Rendering translated text')
    output = await GenerateRender(translated_sentences, mode, task_id, nonce, tgt_lang, img_inpainted, img_rgb, text_regions, render_text_direction_overwrite)

    print(' -- Saving results')
    #if mode == 'ws':
    #    if args.verbose:
    #        # only keep sections in mask
    #        cv2.imwrite(f'result/ws_inmask.png', cv2.cvtColor(img_inpainted, cv2.COLOR_RGB2BGRA) * render_mask)
    #    return dump_image(cv2.cvtColor(output, cv2.COLOR_RGB2RGBA) * render_mask)
    img_pil = dump_image(output, img_alpha)
    img_pil.save(dst_image_name)

    await update_state('finished')

	#print(' -- Saving results')
	#img_pil = dump_image(output, img_alpha)
	#img_pil.save(dst_image_name)
    #
	#if mode == 'web' and task_id:
	#	update_state(task_id, nonce, 'finished')

async def GenerateRender(
	translated_sentences: List[str],
	mode: str,
	task_id: str,
	nonce: str,
	tgt_lang: str,
	img_inpainted: np.ndarray,
	img_rgb: np.ndarray,
	text_regions: List[TextBlock],
	render_text_direction_overwrite: str,
):
    print(' -- Updating Font Size')
    for i, textregion in enumerate(text_regions):
        textregion.font_size = float(translated_sentences[i]['f'])
        textregion.xyxy[0] = int(translated_sentences[i]['x'])
        textregion.xyxy[1] = int(translated_sentences[i]['y'])

    print(' -- Updating translated_sentences')
    translated_sentences = [r['t'] for r in translated_sentences]

    render_mask = np.copy(mask)
    render_mask[render_mask < 127] = 0
    render_mask[render_mask >= 127] = 1
    render_mask = render_mask[:, :, None]

    if tgt_lang == 'ENG' and args.manga2eng:
        from text_rendering import dispatch_eng_render
        dispatch_eng_render_options = [np.copy(img_inpainted), img_rgb, text_regions, translated_sentences]
        if args.font_path:
            font_path = args.font_path.split(',')
            if font_path and font_path[0]:
                dispatch_eng_render_options.append(font_path[0])
        output = await dispatch_eng_render(*dispatch_eng_render_options)
    else:
        output = await dispatch_rendering(np.copy(img_inpainted), args.text_mag_ratio, translated_sentences, text_regions, render_text_direction_overwrite, tgt_lang, args.font_size_offset, render_mask)

    return output

async def infer_safe(
    img: Image.Image,
    mode,
    nonce,
    options = None,
    task_id = '',
    dst_image_name = '',
    update_state = noop,
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
        await update_state('error')

def replace_prefix(s: str, old: str, new: str):
    if s.startswith(old):
        s = new + s[len(old):]
    return s

async def main(mode = 'demo'):
    print(' -- Preload Checks')
    args.image = os.path.expanduser(args.image)
    if not os.path.exists(args.image) and not args.mode.startswith('web'):
        raise FileNotFoundError(args.image)

    if args.use_cuda_limited:
        args.use_cuda = True
    if not torch.cuda.is_available() and args.use_cuda:
        raise Exception('CUDA compatible device could not be found while %s args was set...'
                        % ('--use_cuda_limited' if args.use_cuda_limited else '--use_cuda'))

    print(' -- Loading models')
    os.makedirs('result', exist_ok=True)
    if args.font_path:
        font_path = args.font_path.split(',')
        text_render.prepare_renderer(font_path)
    else:
        text_render.prepare_renderer()
    await prepare_upscaling('waifu2x')
    await prepare_detection(args.detector)
    await prepare_ocr(args.ocr, args.use_cuda)
    await prepare_inpainting(args.inpainter, args.use_cuda)

    if mode == 'demo':
        print(' -- Running in single image demo mode')
        if not args.image:
            print('please provide an image')
            parser.print_usage()
            return
        await infer(Image.open(args.image), mode)

    elif mode == 'web' or mode == 'web2':
        print(' -- Running in web service mode')
        print(' -- Waiting for translation tasks')

        if mode == 'web':
            import subprocess
            import sys
            nonce = crypto_utils.rand_bytes(16).hex()

            extra_web_args = {'stdout':sys.stdout, 'stderr':sys.stderr} if args.log_web else {}
            web_executable = [sys.executable, '-u'] if args.log_web else [sys.executable]
            web_process_args = ['web_main.py', nonce, str(args.host), str(args.port)]
            subprocess.Popen([*web_executable, *web_process_args], **extra_web_args)

        while True:
            try:
                task_id, options = get_task(nonce)

                async def update_state(state):
                    while True:
                        try:
                            requests.post(f'http://{args.host}:{args.port}/task-update-internal', json = {'task_id': task_id, 'nonce': nonce, 'state': state}, timeout = 20)
                            return
                        except Exception:
                            if 'error' in state or 'finished' in state:
                                continue
                            else:
                                break

                if options and 'exit' in options:
                    break
                if task_id:
                    try:
                        print(f' -- Processing task {task_id}')
                        infer_task = asyncio.create_task(infer_safe(Image.open(f'result/{task_id}/input.png'), mode, nonce, options, task_id, None, update_state))
                        asyncio.gather(infer_task)
                    except Exception:
                        import traceback
                        traceback.print_exc()
                        await update_state('error')
                else:
                    await asyncio.sleep(0.1)
            except Exception:
                import traceback
                traceback.print_exc()
    
    elif mode == 'ws':
        print(' -- Running in websocket service mode')

        import io
        import shutil
        import websockets
        import ws_pb2

        WS_SECRET = os.getenv('WS_SECRET', '') # Secret to authenticate with websocket server

        async for websocket in websockets.connect(args.ws_url, extra_headers = { "x-secret": WS_SECRET }, max_size=100_000_000):
            try:
                print(' -- Connected to websocket server')
                async for raw in websocket:
                    msg = ws_pb2.WebSocketMessage()
                    msg.ParseFromString(raw)
                    if msg.WhichOneof('message') == 'new_task':
                        task = msg.new_task

                        if args.verbose:
                            shutil.rmtree(f'result/{task.id}', ignore_errors=True)
                            os.makedirs(f'result/{task.id}', exist_ok=True)

                        options = {
                            'target_language': task.target_language,
                            'detector': task.detector,
                            'direction': task.direction,
                            'translator': task.translator,
                            'size': task.size,
                        }
                        
                        async def update_state(state):
                            msg = ws_pb2.WebSocketMessage()
                            msg.status.id = task.id
                            msg.status.status = state
                            await websocket.send(msg.SerializeToString())

                        print(f' -- Processing task {task.id}')
                        img_pil = await infer_safe(Image.open(io.BytesIO(task.source_image)), mode, None, options, task.id, None, update_state)
                        img = io.BytesIO()
                        img_pil.save(img, format='PNG')
                        img_bytes = img.getvalue()
                        
                        result = ws_pb2.WebSocketMessage()
                        result.finish_task.id = task.id
                        result.finish_task.translation_mask = img_bytes
                        await websocket.send(result.SerializeToString())

                        print(' -- Waiting for translation tasks')

            except Exception:
                import traceback
                traceback.print_exc()

    elif mode == 'batch':
        src = os.path.abspath(args.image)
        if src[-1] == '\\' or src[-1] == '/':
            src = src[:-1]
        dst = args.image_dst or src + '-translated'
        if os.path.exists(dst) and not os.path.isdir(dst):
            print(f'Destination `{dst}` already exists and is not a directory! Please specify another directory.')
            return
        print('Processing image in source directory')
        files = []
        for root, subdirs, files in os.walk(src):
            dst_root = replace_prefix(root, src, dst)
            os.makedirs(dst_root, exist_ok = True)
            for f in files:
                if f.lower() == '.thumb':
                    continue
                filename = os.path.join(root, f)
                dst_filename = replace_prefix(filename, src, dst)
                if os.path.exists(dst_filename):
                    continue
                try:
                    img = Image.open(filename)
                except Exception:
                    pass
                try:
                    print('Processing', filename, '->', dst_filename)
                    await infer(img, 'demo', dst_image_name = dst_filename)
                except Exception:
                    import traceback
                    traceback.print_exc()
                    pass

if __name__ == '__main__':
    try:
        print(args)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(main(args.mode))
    except KeyboardInterrupt:
        print()
