import asyncio
from latentsync import LatentSync

async def main():
    model = LatentSync()
    await model.inference("assets/obama.mp4", "assets/cxk.mp3", "output/obama_cxk.mp4")

if __name__ == "__main__":
    asyncio.run(main())
