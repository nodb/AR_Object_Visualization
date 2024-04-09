import numpy as np
import cv2 as cv

def select_img_from_video(video_file, board_pattern, select_all=False, wait_msec=10, wnd_name='Camera Calibration'):
    # 비디오 열기
    video = cv.VideoCapture(video_file)
    assert video.isOpened()

    # 이미지 선택
    img_select = []
    while True:
        # 비디오에서 이미지 가져오기
        valid, img = video.read()
        if not valid:
            break

        if select_all:
            img_select.append(img)
        else:
            # 이미지 보여주기
            display = img.copy()
            cv.putText(display, f'NSelect: {len(img_select)}', (10, 25), cv.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0))
            cv.imshow(wnd_name, display)

            # 키 이벤트 처리
            key = cv.waitKey(wait_msec)
            if key == ord(' '):            # Space: 이미지 일시 중지, 코너 표시
                complete, pts = cv.findChessboardCorners(img, board_pattern)
                cv.drawChessboardCorners(display, board_pattern, pts, complete)
                cv.imshow(wnd_name, display)
                key = cv.waitKey()
                if key == ord('\r'):       # Enter: 이미지 선택
                    img_select.append(img)
            if key == 27:                  # ESC: 이미지 선택 종료
                break

    cv.destroyAllWindows()
    return img_select

def calib_camera_from_chessboard(images, board_pattern, board_cellsize, K=None, dist_coeff=None, calib_flags=None):
    # 주어진 이미지에서 2D 코너 포인트 찾기
    img_points = []
    for img in images:
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        complete, pts = cv.findChessboardCorners(gray, board_pattern)
        if complete:
            img_points.append(pts)
    assert len(img_points) > 0

    # 체스 보드의 3D 포인트 준비
    obj_pts = [[c, r, 0] for r in range(board_pattern[1]) for c in range(board_pattern[0])]
    obj_points = [np.array(obj_pts, dtype=np.float32) * board_cellsize] * len(img_points) # Must be `np.float32`

    # 카메라 보정
    return cv.calibrateCamera(obj_points, img_points, gray.shape[::-1], K, dist_coeff, flags=calib_flags)

if __name__ == '__main__':
    video_file = 'chessboard.avi'
    board_pattern = (10, 7)
    board_cellsize = 0.025
    board_criteria = cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE + cv.CALIB_CB_FAST_CHECK

    img_select = select_img_from_video(video_file, board_pattern)
    assert len(img_select) > 0, '이미지를 선택하지 않았습니다.'
    rms, K, dist_coeff, rvecs, tvecs = calib_camera_from_chessboard(img_select, board_pattern, board_cellsize)

    # 카메라 보정 결과 출력
    print('카메라 보정 결과')
    print(f'선택된 이미지 수 : {len(img_select)}')
    print(f'RMS 오차 : {rms}')
    print(f'카메라 행렬 (K) = \n{K}')
    print(f'왜곡 계수 (k1, k2, p1, p2, k3, ...) = {dist_coeff.flatten()}')

    # 주어진 비디오와 보정 데이터
    video = cv.VideoCapture(video_file)
    assert video.isOpened(), '주어진 입력을 읽을 수 없습니다: ' + video_file

    # 단순한 AR을 위해 3D 하트 준비
    heart_lower = board_cellsize * np.array([[4, 1.695, 0], [4.3, 0.987, 0], [4.6, 0.69, 0], [4.9, 0.516, 0], [5.2, 0.423, 0], [5.467, 0.399, 0], [5.5, 0.399, 0], [5.8, 0.438, 0], [6.1, 0.54, 0], [6.4, 0.723, 0], [6.7, 1.026, 0], [6.814, 1.203, 0], [6.85, 1.272, 0], [6.934, 1.494, 0], [6.985, 1.959, 0], [6.97, 2.1, 0], [6.892, 2.4, 0], [6.85, 2.505, 0], [6.7, 2.79, 0], [6.553, 3, 0], [6.4, 3.189, 0], [6.1, 3.504, 0], [5.8, 3.78, 0], [5.5, 4.038, 0], [5.2, 4.296, 0], [4.9, 4.563, 0], [4.6, 4.851, 0], [4.3, 5.196, 0], [4.24, 5.28, 0], [4.21, 5.322, 0], [4.18, 5.37, 0], [4.15, 5.418, 0], [4.12, 5.469, 0], [4.09, 5.529, 0], [4.06, 5.592, 0], [4.03, 5.67, 0], [4, 5.805, 0], [3.97, 5.67, 0], [3.94, 5.592, 0], [3.91, 5.529, 0], [3.88, 5.469, 0], [3.85, 5.418, 0], [3.82, 5.37, 0], [3.79, 5.322, 0], [3.76, 5.28, 0], [3.7, 5.196, 0], [3.4, 4.851, 0], [3.1, 4.563, 0], [2.8, 4.296, 0], [2.5, 4.038, 0], [2.2, 3.78, 0], [1.9, 3.504, 0], [1.6, 3.189, 0], [1.447, 3, 0], [1.3, 2.79, 0], [1.15, 2.505, 0], [1.108, 2.4, 0], [1.03, 2.1, 0], [1.015, 1.959, 0], [1.066, 1.494, 0], [1.15, 1.272, 0], [1.186, 1.203, 0], [1.3, 1.026, 0], [1.6, 0.723, 0], [1.9, 0.54, 0], [2.2, 0.438, 0], [2.5, 0.399, 0], [2.533, 0.399, 0], [2.8, 0.423, 0], [3.1, 0.516, 0], [3.4, 0.69, 0], [3.7, 0.987, 0], [4, 1.695, 0]
    ])
    heart_upper = board_cellsize * np.array([[4, 1.695, -2], [4.3, 0.987, -2], [4.6, 0.69, -2], [4.9, 0.516, -2], [5.2, 0.423, -2], [5.467, 0.399, -2], [5.5, 0.399, -2], [5.8, 0.438, -2], [6.1, 0.54, -2], [6.4, 0.723, -2], [6.7, 1.026, -2], [6.814, 1.203, -2], [6.85, 1.272, -2], [6.934, 1.494, -2], [6.985, 1.959, -2], [6.97, 2.1, -2], [6.892, 2.4, -2], [6.85, 2.505, -2], [6.7, 2.79, -2], [6.553, 3, -2], [6.4, 3.189, -2], [6.1, 3.504, -2], [5.8, 3.78, -2], [5.5, 4.038, -2], [5.2, 4.296, -2], [4.9, 4.563, -2], [4.6, 4.851, -2], [4.3, 5.196, -2], [4.24, 5.28, -2], [4.21, 5.322, -2], [4.18, 5.37, -2], [4.15, 5.418, -2], [4.12, 5.469, -2], [4.09, 5.529, -2], [4.06, 5.592, -2], [4.03, 5.67, -2], [4, 5.805, -2], [3.97, 5.67, -2], [3.94, 5.592, -2], [3.91, 5.529, -2], [3.88, 5.469, -2], [3.85, 5.418, -2], [3.82, 5.37, -2], [3.79, 5.322, -2], [3.76, 5.28, -2], [3.7, 5.196, -2], [3.4, 4.851, -2], [3.1, 4.563, -2], [2.8, 4.296, -2], [2.5, 4.038, -2], [2.2, 3.78, -2], [1.9, 3.504, -2], [1.6, 3.189, -2], [1.447, 3, -2], [1.3, 2.79, -2], [1.15, 2.505, -2], [1.108, 2.4, -2], [1.03, 2.1, -2], [1.015, 1.959, -2], [1.066, 1.494, -2], [1.15, 1.272, -2], [1.186, 1.203, -2], [1.3, 1.026, -2], [1.6, 0.723, -2], [1.9, 0.54, -2], [2.2, 0.438, -2], [2.5, 0.399, -2], [2.533, 0.399, -2], [2.8, 0.423, -2], [3.1, 0.516, -2], [3.4, 0.69, -2], [3.7, 0.987, -2], [4, 1.695, -2]
    ])

    # 체스보드의 3D 점 준비
    obj_points = board_cellsize * np.array([[c, r, 0] for r in range(board_pattern[1]) for c in range(board_pattern[0])])

    # 왜곡 보정 실행
    show_rectify = False
    map1, map2 = None, None
    while True:
        # 비디오에서 이미지 읽기
        valid, img = video.read()
        if not valid:
            break

        # 기하학적 왜곡 보정 (대안: `cv.undistort()`)
        info = "Original"
        if show_rectify:
            if map1 is None or map2 is None:
                map1, map2 = cv.initUndistortRectifyMap(K, dist_coeff, None, None, (img.shape[1], img.shape[0]), cv.CV_32FC1)
            img = cv.remap(img, map1, map2, interpolation=cv.INTER_LINEAR)
            info = "Distortion Correction"
        cv.putText(img, info, (10, 45), cv.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0))
        
        # 카메라 자세 추정
        complete, pts = cv.findChessboardCorners(img, board_pattern, board_criteria)
        if complete:
            ret, rvec, tvec = cv.solvePnP(obj_points, pts, K, dist_coeff)

            # 이미지에 박스 그리기
            line_lower, _ = cv.projectPoints(heart_lower, rvec, tvec, K, dist_coeff)
            line_upper, _ = cv.projectPoints(heart_upper, rvec, tvec, K, dist_coeff)
            cv.polylines(img, [np.int32(line_lower)], True, (255, 255, 255), 2)
            cv.polylines(img, [np.int32(line_upper)], True, (0, 0, 255), 3)
            for b, t in zip(line_lower, line_upper):
                cv.line(img, np.int32(b.flatten()), np.int32(t.flatten()), (255, 255, 255), 1)

            # 카메라 위치 출력
            R, _ = cv.Rodrigues(rvec) # 대체) `scipy.spatial.transform.Rotation`
            p = (-R.T @ tvec).flatten()
            info = f'XYZ: [{p[0]:.3f} {p[1]:.3f} {p[2]:.3f}]'
            cv.putText(img, info, (10, 25), cv.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0))

        # 이미지 보여주고 키 이벤트 처리
        cv.imshow("Geometric Distortion Correction", img)
        key = cv.waitKey(10)
        if key == ord(' '):     # Space: 일시 중지
            key = cv.waitKey()
        if key == 27:           # ESC: 종료
            break
        elif key == ord('\t'):  # Tab: 보정 모드 전환
            show_rectify = not show_rectify

    video.release()
    cv.destroyAllWindows()
