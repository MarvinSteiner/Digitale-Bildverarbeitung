import cv2
import numpy as np
from copy import deepcopy
from matplotlib import pyplot as plt

from . import Algorithm


class InvisCloak(Algorithm):

    def __init__(self):
        self.image_stack = list()
        self.background = None
        self.capture_background = False

        # FIX: Histogramm nur 1x anzeigen
        self.hist_done = False

    def process(self, img):

        """ 2.1 Vorverarbeitung """

        plotNoise = False
        if plotNoise:
            self._plotNoise(img, "Rauschen vor Korrektur")

        img = self._211_Rauschreduktion(img)

        if plotNoise:
            self._plotNoise(img, "Rauschen nach Korrektur")

        """ 2.1.2 HistogrammSpreizung """
        img = self._212_HistogrammSpreizung(img)

        """ 2.2 Farbanalyse (NUR EINMAL AUSFÜHREN!) """
        if not self.hist_done:
            self._221_RGB(img)
            self._222_HSV(img)
            self.hist_done = True

        """ 2.3 """
        img = self._23_SegmentUndBildmodifizierung(img)

        return img

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONUP:
            print("A Mouse click happend! at position", x, y)
            self.capture_background = True

    def _plotNoise(self, img, name: str):
        height, width = np.array(img.shape[:2])
        centY = (height / 2).astype(int)
        centX = (width / 2).astype(int)

        cutOut = 5
        tmpImg = deepcopy(img)
        tmpImg = tmpImg[centY - cutOut:centY + cutOut,
                        centX - cutOut:centX + cutOut, :]

        outSize = 500
        tmpImg = cv2.resize(tmpImg, (outSize, outSize),
                            interpolation=cv2.INTER_NEAREST)

        cv2.imshow(name, tmpImg)
        cv2.waitKey(1)

    def _211_Rauschreduktion(self, img):

        N = 3
        max_image_stack_length = N + 1

        self.image_stack.append(img)
        if len(self.image_stack) > max_image_stack_length:
            self.image_stack = self.image_stack[-max_image_stack_length:]

        stack = np.stack(self.image_stack, axis=0)

        sum_img = np.sum(stack.astype(np.uint32), axis=0)

        num_images = len(self.image_stack)

        mean_img = sum_img // num_images

        return mean_img.astype(np.uint8)

    def _212_HistogrammSpreizung(self, img):

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        v = cv2.normalize(
            v,
            None,
            alpha=0,
            beta=255,
            norm_type=cv2.NORM_MINMAX
        )

        hsv = cv2.merge((h, s, v))

        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def _221_RGB(self, img):

        b_hist = cv2.calcHist([img], [0], None, [256], [0, 256])
        g_hist = cv2.calcHist([img], [1], None, [256], [0, 256])
        r_hist = cv2.calcHist([img], [2], None, [256], [0, 256])

        plt.figure("RGB Histogramm", figsize=(8, 4))
        plt.clf()

        plt.plot(b_hist, color="b", label="Blue")
        plt.plot(g_hist, color="g", label="Green")
        plt.plot(r_hist, color="r", label="Red")

        plt.title("RGB Histogramm")
        plt.xlabel("Intensität (0–255)")
        plt.ylabel("Pixelanzahl")
        plt.xlim([0, 256])
        plt.legend()

        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.001)

        return r_hist, g_hist, b_hist

    def _222_HSV(self, img):

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        h_hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
        s_hist = cv2.calcHist([hsv], [1], None, [256], [0, 256])
        v_hist = cv2.calcHist([hsv], [2], None, [256], [0, 256])

        plt.figure("HSV Histogramm", figsize=(8, 4))
        plt.clf()

        plt.plot(h_hist, label="Hue (H)")
        plt.plot(s_hist, label="Saturation (S)")
        plt.plot(v_hist, label="Value (V)")

        plt.title("HSV Histogramm")
        plt.xlabel("Intensität")
        plt.ylabel("Pixelanzahl")

        plt.legend()
        plt.tight_layout()

        plt.show(block=False)
        plt.pause(0.001)

        return h_hist, s_hist, v_hist

    def _23_SegmentUndBildmodifizierung(self, img):
        """
            Hier steht Ihr Code zu Aufgabe 2.3.1 (StatischesSchwellwertverfahren)
            - Binärmaske erstellen
        """
        #von BGR in HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        #Rot Range 1
        lower_red1 = np.array([0, 50, 40])
        upper_red1 = np.array([10, 255, 255])

        #Rot Range 2
        lower_red2 = np.array([170, 50, 40])
        upper_red2 = np.array([180, 255, 255])

        #Binärmaske erstellen
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = mask1 + mask2

        """
            Hier steht Ihr Code zu Aufgabe 2.3.2 (Binärmaske)
            - Binärmaske optimieren mit Opening/Closing
            - Wahl größte zusammenhängende Region
        """
        #5x5 Matrix Kernel (nur 1)
        kernel = np.ones((5, 5), np.uint8)

        #Opening (Erosion + Dilatation)
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            kernel,
            iterations=2
        )

        #Closing (Dilatation + Erosion)
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=2
        )

        #Außenlinie (Kontur) finden
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL, #nur äußere Konturen, keine inneren Konturen
            cv2.CHAIN_APPROX_SIMPLE
        )

        #Leere Maske erstellen für Speicherung der größten Region
        largest_mask = np.zeros_like(mask)

        if len(contours) > 0: #Bugfix, falls kein Umhang sichtbar ist
            #Fläche jeder Kontur berechnen
            largest_contour = max(contours, key=cv2.contourArea)

            #größte Kontur in der leeren Bitmaske einfärben
            cv2.drawContours(
                largest_mask,
                [largest_contour],
                -1,
                255,
                thickness=cv2.FILLED #Umhang ist ganze Fläche, nicht nur Außenlinie
            )

            #Annahme: Umhang ist größte rote Kontur im Bild (größtes Objekt flächenmäßig)
            mask = largest_mask

        """
            Hier steht Ihr Code zu Aufgabe 2.3.1 (Bildmodifizerung)
            - Hintergrund mit Mausklick definieren
            - Ersetzen des Hintergrundes
        """
        #Hintergrund aufnehmen
        if self.capture_background:
            self.background = img.copy() #copy sonst jedes neue Kamerabild = neuer Hintergrund
            self.capture_background = False

        #Falls noch kein Hintergrund aufgenommen wurde
        if self.background is None:
            return img

        #Maske invertieren
        mask_inv = cv2.bitwise_not(mask)

        #Vordergrund extrahieren (alles außer Umhang)
        foreground = cv2.bitwise_and(
            img,
            img,
            mask=mask_inv
        )

        #Hintergrund extrahieren aus Momentaufnahme beim Mausklick (Bereich des Umhangs)
        background_part = cv2.bitwise_and(
            self.background,
            self.background,
            mask=mask
        )

        #Vorder- und  Hintergrund zusammenfügen
        result = cv2.add(
            foreground,
            background_part
        )

        return result