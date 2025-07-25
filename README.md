# UniAli v1.xx - dev branch
![](Images/logo.png)

Это была моя третья попытка примотать легковесную Alibaba LLM к UWP. 
Частично-удачная, но недодуманная. Токенайзер временно взят от ChatGPT. Это не то. См. папку research\ и пояснения Deepseek, как быть.
В сорцах в Assets тестовая модель model_q4f16.onn отсутствует (я просто брал готовую тут [model_q4f16.onnx](https://huggingface.co/onnx-community/Qwen3-1.7B-ONNX) и еще ничего не квартировал/конвертил для большей совместимости с Microsoft QNNX) . Я её специально удалил, ибо весит она под 1,33 гига …. зато в какой-то момент токенайзер её даже "увидел" чуток). Что ж... видимо, на третей попытке придется через "взрослые" АИ вроде ChatGPT прогнать это творение... не знаю... нужны железные мозги, чтоб в это сербезно вникнуть! Вообщем, пока как то так. 


## Архитектура
- UniALi (NET Standard 2, remastered)
- TiktokenSharp (UWP, remastered)

## Environment
- Windows SDK Target Version: 10.0.19041.0
- Windows SDK Min. Version:   10.0.16299.0 

## Заключение
Дело идет. Медленно, т.к. че то ноль активности народа.

## Credits / Ссыли
- Товарищу Deleted с форума 4PDA за предложение создать эту шарманку для старых добрых винфонов
- https://huggingface.co/onnx-community/Qwen3-0.6B-ONNX Где-то тут водятся творения от энтузиастов и Ali-бабы
- https://4pda.to/forum/index.php?showtopic=1107793#entry138245556 Собственно мой случайный диалог c Deleted тут.
- https://github.com/microsoft/onnxruntime ONNX Runtime: кросс-платформенный суперпупернавороченный ML-аксселератор/тренер ML-моделей
- 

## ..
AS IS. RnD only. No support. DIY

## .
[m][e] 2025
