# QRCodeStore
Store arbitrary files as sequences of QR codes. This script ingests a file, chunks it into sizes small enough to fit into QR codes, concurrently generates QR codes for those chunks, combines them in separate RGB channels of images, and then combines those colored images into gif and webp sequence formats.
