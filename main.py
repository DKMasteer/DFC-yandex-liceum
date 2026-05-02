from inference.inferencer import DFC_Activator

inf = DFC_Activator(model_type='dfc-3.4')

tests = [
                    '/Users/daniilkrasnov/Desktop/YL_Project_Solution/photos/1.jpg', # Дипфейк
                    '/Users/daniilkrasnov/Desktop/YL_Project_Solution/photos/deepfake_photo.png', # Дипфейк
                    '/Users/daniilkrasnov/Desktop/YL_Project_Solution/photos/not_deepfake_photo.jpg', # Не дипфейк
                    '/Users/daniilkrasnov/Desktop/YL_Project_Solution/photos/0.jpg' # Не дипфейк
                    ]

answers = [True, True, False, False]

for test, answer in zip(tests, answers):
    is_deepfake, prob = inf(test)

    if is_deepfake == answer:
        print(f"Тест пройден успешно! Вероятность дипфейка: {prob}")
    else:
        print(f"Тест провален! Вероятность дипфейка: {prob}, ожидалось: {answer}")