"""English edition of the existing deck, preserving its images and layout."""
from copy import deepcopy
from pathlib import Path
import json
from PIL import Image
from tools.slide_design import fit, width, write_pdf

# Element indices retain separately styled text runs from the original deck.
TEXT = {
 'cover': {1: 'Pedestrian detection\nwith 3 types of data fusion'},
 'objective-comparison': {0: 'Objective: compare three fusion methods', 9: 'Do the network and fine-tuning help', 10: 'detect more pedestrians', 11: 'compared with non-learning-based data fusion?'},
 'dataset-inputs': {0: 'Dataset: AWMM-100K and two image types', 5: 'Visible: colour camera', 6: 'Thermal: infrared camera', 7: 'AWMM-100K: pretraining. Images shown: LLVIP, used for our fine-tuning.'},
 'rain-input': {0: 'Artificial rain in our experiment', 1: 'In our experiment, artificial rain is added only to the colour camera images.', 4: 'Goal: compare the 3 types of data fusion using', 5: 'the image with rain and the infrared camera image'},
 'complete-pipeline': {0: 'The complete pipeline:', 4: 'Visible image\nwith rain', 6: 'Thermal image', 8: 'Fusion process\n(method 1/2/3)', 10: 'Fused\nimage', 14: 'Detected\npedestrians'},
 'fixed-average': {3: 'Rule: average the corresponding pixels of the two images', 5: 'Fused image =', 6: 'Visible + Thermal', 9: 'Each pixel receives half its contribution from\neach image.\nThe weights are the same everywhere:\nthe fusion does not learn.'},
 'neural-fusion': {3: '• Pretrained network used:', 5: '• Dataset used: AWMM-100K with 1,000 image pairs', 6: '• AMG-Fuse processes visible and thermal inputs to directly produce an image', 9: '1. Extracts features', 10: 'Finds edges and contrasts\nin the two inputs.', 12: '2. Combines them', 13: 'Combines information\nthrough attention.', 15: '3. Reconstructs', 16: 'Produces an initial', 17: 'combined image'},
 'neural-fusion-weights': {0: 'How the network guides fusion', 1: 'The code takes R, then computes α by comparing it with the two inputs.', 3: '• R closer to visible → α > 0.5 → greater visible contribution.', 4: '• R closer to thermal → α < 0.5 → greater thermal contribution.', 6: 'Fused image = α × Visible + (1 − α) × Thermal', 7: 'Fixed average: α = 0.5 everywhere. Here, the weight varies with the scene and position.'},
 'fine-tuning-method': {3: '• We start with the same AMG-Fuse and adapt only the', 4: 'last layer', 6: 'LLVIP dataset', 7: ': 100 image pairs', 8: '• 3 epochs:', 9: '2,592', 10: 'updated weights', 12: '1. Detection', 13: 'Fused image → YOLO →\npedestrian boxes.', 15: '2. Error', 16: 'Comparison with the ground\ntruth boxes in the dataset.', 18: '3. Update', 19: 'The error updates only\nthe final fusion layer.'},
 'yolo-threshold': {0: 'YOLO: from fused images to pedestrians', 1: 'Each fused image from the three methods is fed separately into the same YOLO.', 4: 'Fused image → YOLO → pedestrian boxes + scores', 5: 'Threshold = minimum score required to show a detection.', 7: 'Higher threshold ↑', 8: 'Fewer boxes shown.\nMay reduce false alarms,\nbut also miss real pedestrians.', 10: 'Lower threshold ↓', 11: 'More boxes shown.\nMay recover pedestrians,\nbut also add false alarms.'},
 'demo-transition': {0: 'Interface', 3: 'We now compare the three methods on the same', 5: 'image pairs', 6: 'from the LLVIP test set using the', 7: 'SAME YOLO'},
 'challenges': {0: 'Three challenges and how they were solved', 1: 'Annotated data, CPU cost and accurate detection evaluation.', 3: 'Data', 4: 'Reproducible subset, ground truth annotations,\nverified files.', 6: 'Cached features; only the final layer is updated.', 8: 'Detector prepared first, then frozen.\nGround truth annotations and a new test set.'},
 'thanks': {0: 'Thank you for your attention', 1: 'Questions and discussion', 3: 'Learning to fuse:\nhow much does it help detection?'},
}

NOTES = {
 'cover': 'The goal is to find out whether adapting visible–thermal fusion helps detect more pedestrians when rain degrades the visible image. We use real images and simulated rain. We compare fixed averaging, AMG-Fuse-guided fusion before adaptation and the same fusion after adaptation. The detector remains identical. This version replaces the earlier small experiment with pseudo references.',
 'objective-comparison': 'We compare three ways of combining the same visible and thermal images with simulated rain: fixed averaging, a pretrained network and the same network after fine-tuning. All methods are evaluated on the same 50 test images with the same frozen detector. We measure detected pedestrians and false positives: improvement must be verified rather than assumed.',
 'dataset-inputs': 'We use LLVIP, a dataset of visible and thermal image pairs of the same scene, mainly in low light. The images shown are the original pair from the same scene: a standard colour camera on the left and an infrared thermal camera on the right. They are already aligned and have annotated pedestrian boxes. Of the full dataset of 15,488 pairs, we use 170: 100 to update the weights, 20 for validation and 50 for testing. We add artificial rain only to the visible image; the thermal image remains unchanged. LLVIP is not a real rain dataset.',
 'rain-input': 'We show the same colour image as on the previous slide after applying the artificial rain used in the experiment. It is not a photograph taken in real rain. The thermal image of the same scene remains unchanged. The goal is to compare the three types of data fusion — fixed averaging, a pretrained network and the network after fine-tuning — using the same pair of inputs: the colour image with simulated rain and the infrared camera image. The detector remains identical in all three comparisons.',
 'complete-pipeline': 'The visible image with rain and the thermal image enter the fusion process. The fusion process block represents one of the three methods being compared: fixed averaging, pretrained fusion or fusion after fine-tuning. Each method produces one fused image, which the same YOLO detector analyses to detect pedestrians.',
 'fixed-average': 'This method combines already aligned images without learning from data. At each position, it takes the normalised visible and thermal values and averages them: F = 0.5 V + 0.5 T. The single thermal channel is replicated to combine it with the three colour channels. The example of 100 and 200 averaging to 150 uses illustrative values on a common scale, not temperatures in degrees. The weights remain 50% and 50% in every region and scene, even when an input is degraded. There are no fusion weights to train. The fused image is then fed to the same YOLO detector used by the other two methods. “Non-learning-based” refers to fusion, not the detector.',
 'neural-fusion': 'We use AMG-Fuse’s MSAKNet, loading the authors’ Haze checkpoint, already trained for visible–infrared fusion in haze. We do not start from random weights and perform no fine-tuning of our own in this second method. The network contains convolutions for local structures and Transformer blocks with attention to relate features. The visible, thermal and fusion branches exchange information at multiple scales. Features are learned numerical maps, not pedestrian labels. To run on the CPU, we reduce both inputs to 40 × 32 pixels, extract features before the final output_f convolution and apply this original layer to obtain a small RGB reconstruction. The next slide explains how this reconstruction guides fusion of the 320 × 256 inputs. This is an AMG-Fuse-guided variant, distinct from its native full-resolution output.',
 'neural-fusion-weights': 'The network does not directly produce the alpha map used in our final blend. First, output_f reconstructs a result R at 40 × 32 pixels. At each position, we compute the mean squared distance across the three channels between R and the reduced visible input, dV, and between R and the reduced thermal input, dT. The thermal input is expanded to three channels. We define alpha = sigmoid(4 × (dT − dV)). If R is closer to the visible input, dV is smaller than dT and alpha exceeds 0.5; if it is closer to the thermal input, alpha is below 0.5. The sigmoid keeps coefficients between zero and one. We enlarge the map to 320 × 256 with bilinear interpolation and compute F = alpha V + (1 − alpha) T, using the higher-resolution inputs to retain detail. Fixed averaging corresponds to alpha always equal to 0.5; here alpha depends on the inputs and varies by position. For example, alpha = 0.8 means 80% visible and 20% thermal: this illustrates its interpretation and is not a measured value. Alpha is not a reliability probability. Any detection benefit must be measured with YOLO. This procedure is our AMG-Fuse-guided variant, not its native full-resolution output.',
 'fine-tuning-method': 'Fine-tuning starts from pretrained weights, not a random network. We use 100 LLVIP pairs with simulated rain on the visible input and ground truth pedestrian annotations for 3 epochs. Fusion produces an image and YOLO proposes pedestrians. The loss compares classification and location with ground truth boxes; backpropagation passes through the detector and updates only the 2,592 weights of AMG-Fuse’s final layer. The rest of the fusion network and the entire detector remain frozen. A small regularisation term limits deviation of the map from its initial version. The 20 validation pairs recommend a checkpoint; we nevertheless compare the weights actually updated after three epochs on the 50 test pairs. The common detector is prepared before the comparison and remains identical across the three methods. Validation recommends the original weights: fine-tuning is not presented as a guaranteed improvement.',
 'yolo-threshold': 'Images produced by fixed averaging, pretrained fusion and fusion after fine-tuning are analysed separately by the same YOLO detector with identical parameters. YOLO searches for pedestrians and proposes rectangles with confidence scores. The threshold is the minimum score required to retain a detection: at 0.30, a proposal scoring 0.75 is shown and one scoring 0.20 is hidden. Raising the threshold shows fewer proposals: it may remove false alarms but also lose correct pedestrians. Lowering it shows more proposals: it may recover pedestrians but also add false alarms. The score is not a calibrated probability. In the demo, the slider changes only the displayed boxes; aggregate results are computed at the shared threshold of 0.30 chosen on validation.',
 'demo-transition': 'We now compare the three methods on the same 50 test pairs, separate from training, using the same YOLO. The local interface is available at http://127.0.0.1:64001 after running AVVIA_DEMO.cmd. We first show one scene with both inputs, the three fused images and detections, then the aggregate results.',
 'challenges': 'The first challenge was dataset cost: the full archive is about 4 GB, so we read its remote index and downloaded a subset with XML annotations, verifying each file. The final experiment uses 170 pairs, in addition to diagnostic pairs excluded from the final test. The second challenge was the network’s CPU cost: we froze and cached features, adapted only the final layer and preserved input detail through the fusion map. The third challenge was the generic detector, poorly suited to night-time and thermal scenes: we prepare a common head on the 100 pairs, then freeze it for comparison. Ground truth boxes and a new separate test set let us measure recall, precision, AP50 and false positives.',
 'thanks': 'The final message is that fusion can be adapted using a concrete detection objective while keeping the detector fixed. We used 100 pairs for fine-tuning and 50 for testing, with simulated rain. The measured change is +11 correctly detected pedestrians and +8.59 percentage points of recall. This must be interpreted alongside false positives and AP50. Sources and limitations are in the report, without a final references slide.',
}

def english_slides(slides):
    result = deepcopy(slides)
    for number, s in enumerate(result, 1):
        key = s['layout']
        if key in ('challenges', 'lessons-learned', 'thanks'):
            from tools.closing_slides import closing_slide
            s.update(closing_slide(key, number, 'en'))
            continue
        if key == 'performance-comparison':
            from tools.performance_slide import performance_slide
            replacement = performance_slide(Path(__file__).resolve().parents[1]/'presentazione', 'en')
            s.update(replacement)
            continue
        for i, value in TEXT[key].items():
            a = s['elements'][i]
            a['text'], a['size'] = fit(value, a['w'], a['h'], a['size'], a['bold'])
        s['note'] = NOTES[key]
        s['title'] = TEXT[key].get(0, s['title'])
        s['subtitle'] = TEXT[key].get(1, s['subtitle']) if key != 'cover' else ''
        if key == 'cover':
            s['title'] = 'Pedestrian detection with 3 types of data fusion'
        # Reposition adjacent styled runs using the translated font metrics.
        groups = {
            'objective-comparison': [(9, 10)],
            'neural-fusion': [(3, 4), (6, 7), (17, 18)],
            'fine-tuning-method': [(3, 4), (5, 6, 7), (8, 9, 10)],
            'demo-transition': [(3, 4, 5), (6, 7)],
        }.get(key, [])
        for indices in groups:
            runs = [s['elements'][i] for i in indices]
            for a in runs:
                a['text'] = a['text'].replace('\n', ' ')
            sizes = [a['size'] for a in runs]
            size = min(sizes)
            def lengths():
                return [width(a['text'], size, a['bold']) / 72 for a in runs]
            available = 11.3 if key != 'neural-fusion' or indices != (17, 18) else 3.46
            while sum(lengths()) + .12 * (len(runs) - 1) > available:
                size -= 1
            widths = lengths()
            total = sum(widths) + .12 * (len(runs) - 1)
            x = (13.333 - total) / 2 if key in ('objective-comparison', 'demo-transition') else 8.92 + (3.46-total)/2 if indices == (17, 18) else 1
            for a, w in zip(runs, widths):
                a.update(x=x, w=w+.04, size=size)
                x += w + .12
    return result

def overview(slides, out):
    sheet = Image.new('RGB', (1600, 225*((len(slides)+3)//4)), '#edf3f8')
    for i in range(len(slides)):
        with Image.open(out/f'anteprima_{i+1:02d}.png') as source:
            im = source.convert('RGB'); im.thumbnail((400,225))
            sheet.paste(im, ((i%4)*400, (i//4)*225))
    sheet.save(out/'panoramica.png')

def write_english(slides, out):
    out = out/'english'
    out.mkdir(exist_ok=True)
    translated = english_slides(slides)
    write_pdf(translated, out)
    (out/'slides.json').write_text(json.dumps(translated, ensure_ascii=False, indent=2), encoding='utf-8')
    guide = 'WEATHERFUSION — SPEAKER NOTES\n\n'
    for i, s in enumerate(translated, 1):
        guide += f"SLIDE {i} — {s['title']}\n\n{s['note']}\n\n"
    (out/'Speaker_notes.txt').write_text(guide, encoding='utf-8')
    overview(translated, out)
