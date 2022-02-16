package main

import (
	"errors"
	"fmt"
	"time"
	"unsafe"

	"github.com/charliehorse55/go-opencl/cl"
)

// #include "cbdrank.h"
import "C"

//go:embed rank.cl
var kernel_source string

func getFirstOpenCLDevice() (*cl.Device, error) {
	platforms, err := cl.GetPlatforms()
	if err != nil {
		return nil, err
	}

	if len(platforms) == 0 {
		return nil, errors.New("No OpenCL platforms found")
	}

	platform := platforms[0]

	devices, err := cl.GetDevices(platform, cl.DeviceTypeAll)
	if err != nil {
		return nil, err
	}

	if len(devices) == 0 {
		return nil, errors.New("No OpenCL devices found")
	}

	device := devices[0]

	return device, nil
}

func build_program_and_get_kernel(device *cl.Device, clContext *cl.Context) (*cl.Kernel, error) {
	program, err := clContext.CreateProgramWithSource([]string{kernel_source})
	if err != nil {
		return nil, err
	}

	err = program.BuildProgram([]*cl.Device{device}, "")
	if err != nil {
		return nil, err
	}

	return program.CreateKernel("calculate_rank")
}

func EnqueueWriteBufferUint32(q *cl.CommandQueue, buffer *cl.MemObject, blocking bool, offset int, data []uint32, eventWaitList []*cl.Event) (*cl.Event, error) {
	dataPtr := unsafe.Pointer(&data[0])
	dataSize := int(unsafe.Sizeof(data[0])) * len(data)
	return q.EnqueueWriteBuffer(buffer, blocking, offset, dataSize, dataPtr, eventWaitList)
}

func EnqueueWriteBufferUint64(q *cl.CommandQueue, buffer *cl.MemObject, blocking bool, offset int, data []uint64, eventWaitList []*cl.Event) (*cl.Event, error) {
	dataPtr := unsafe.Pointer(&data[0])
	dataSize := int(unsafe.Sizeof(data[0])) * len(data)
	return q.EnqueueWriteBuffer(buffer, blocking, offset, dataSize, dataPtr, eventWaitList)
}

func EnqueueWriteBufferFloat64(q *cl.CommandQueue, buffer *cl.MemObject, blocking bool, offset int, data []float64, eventWaitList []*cl.Event) (*cl.Event, error) {
	dataPtr := unsafe.Pointer(&data[0])
	dataSize := int(unsafe.Sizeof(data[0])) * len(data)
	return q.EnqueueWriteBuffer(buffer, blocking, offset, dataSize, dataPtr, eventWaitList)
}

func EnqueueReadBufferFloat64(q *cl.CommandQueue, buffer *cl.MemObject, blocking bool, offset int, data []float64, eventWaitList []*cl.Event) (*cl.Event, error) {
	dataPtr := unsafe.Pointer(&data[0])
	dataSize := int(unsafe.Sizeof(data[0])) * len(data)
	return q.EnqueueReadBuffer(buffer, blocking, offset, dataSize, dataPtr, eventWaitList)
}

func SetArgFloat64(k *cl.Kernel, index int, val float64) error {
	return k.SetArgUnsafe(index, int(unsafe.Sizeof(val)), unsafe.Pointer(&val))
}

func get_links_start_index(cidsSize uint64, links []uint32) (uint64, []uint64) {
	startIndex := make([]uint64, cidsSize)
	var index uint64
	for i := uint64(0); i < cidsSize; i++ {
		startIndex[i] = index
		index += uint64(links[i])
	}
	return index, startIndex
}

func calculate_rank(clContext *cl.Context, queue *cl.CommandQueue, kernel *cl.Kernel,
	stakes []uint64,
	cidsCount int64,
	inLinksCount, outLinksCount []uint32,
	outLinksIns, inLinksOuts []uint64,
	inLinksUsers, outLinksUsers []uint64,
	dampingFactor, tolerance float64) ([]float64, []float64, []float64, error) {
	stakesSize := len(stakes)
	cidsSize := len(inLinksCount)
	linksSize := len(inLinksOuts)

	rank := make([]float64, cidsCount)
	entropy := make([]float64, cidsCount)
	luminosity := make([]float64, cidsCount)
	karma := make([]float64, stakesSize)

	start := time.Now()

	// ins
	inStakesBuf, err := clContext.CreateEmptyBuffer(cl.MemReadOnly, 8*stakesSize)
	if err != nil {
		return nil, nil, nil, err
	}
	inInLinksCountBuf, err := clContext.CreateEmptyBuffer(cl.MemReadOnly, 4*cidsSize)
	if err != nil {
		return nil, nil, nil, err
	}
	inOutLinksCountBuf, err := clContext.CreateEmptyBuffer(cl.MemReadOnly, 4*cidsSize)
	if err != nil {
		return nil, nil, nil, err
	}
	inOutLinksInsBuf, err := clContext.CreateEmptyBuffer(cl.MemReadOnly, 8*linksSize)
	if err != nil {
		return nil, nil, nil, err
	}
	inInLinksOutsBuf, err := clContext.CreateEmptyBuffer(cl.MemReadOnly, 8*linksSize)
	if err != nil {
		return nil, nil, nil, err
	}

	_, inLinksStartIndex := get_links_start_index(uint64(cidsSize), inLinksCount)
	_, outLinksStartIndex := get_links_start_index(uint64(cidsSize), outLinksCount)

	inInLinksStartIndexBuf, err := clContext.CreateEmptyBuffer(cl.MemReadOnly, 8*cidsSize)
	if err != nil {
		return nil, nil, nil, err
	}
	inOutLinksStartIndexBuf, err := clContext.CreateEmptyBuffer(cl.MemReadOnly, 8*cidsSize)
	if err != nil {
		return nil, nil, nil, err
	}

	// outs
	outRankBuf, err := clContext.CreateEmptyBuffer(cl.MemWriteOnly, 8*int(cidsCount))
	if err != nil {
		return nil, nil, nil, err
	}
	outEntropyBuf, err := clContext.CreateEmptyBuffer(cl.MemWriteOnly, 8*int(cidsCount))
	if err != nil {
		return nil, nil, nil, err
	}
	outKarmaBuf, err := clContext.CreateEmptyBuffer(cl.MemWriteOnly, 8*int(cidsCount))
	if err != nil {
		return nil, nil, nil, err
	}
	outLuminocityBuf, err := clContext.CreateEmptyBuffer(cl.MemWriteOnly, 8*int(cidsCount))
	if err != nil {
		return nil, nil, nil, err
	}

	// copy data to buffers
	_, err = EnqueueWriteBufferUint64(queue, inStakesBuf, true, 0, stakes, nil)
	if err != nil {
		return nil, nil, nil, err
	}
	_, err = EnqueueWriteBufferUint32(queue, inInLinksCountBuf, true, 0, inLinksCount, nil)
	if err != nil {
		return nil, nil, nil, err
	}
	_, err = EnqueueWriteBufferUint32(queue, inOutLinksCountBuf, true, 0, outLinksCount, nil)
	if err != nil {
		return nil, nil, nil, err
	}
	_, err = EnqueueWriteBufferUint64(queue, inOutLinksInsBuf, true, 0, outLinksIns, nil)
	if err != nil {
		return nil, nil, nil, err
	}
	_, err = EnqueueWriteBufferUint64(queue, inInLinksOutsBuf, true, 0, inLinksOuts, nil)
	if err != nil {
		return nil, nil, nil, err
	}
	_, err = EnqueueWriteBufferUint64(queue, inInLinksStartIndexBuf, true, 0, inLinksStartIndex, nil)
	if err != nil {
		return nil, nil, nil, err
	}
	_, err = EnqueueWriteBufferUint64(queue, inOutLinksStartIndexBuf, true, 0, outLinksStartIndex, nil)
	if err != nil {
		return nil, nil, nil, err
	}

	// kernel args
	kernel.SetArgBuffer(0, inStakesBuf)
	kernel.SetArgBuffer(1, inInLinksCountBuf)
	kernel.SetArgBuffer(2, inOutLinksCountBuf)
	kernel.SetArgBuffer(3, inOutLinksInsBuf)
	kernel.SetArgBuffer(4, inInLinksOutsBuf)
	kernel.SetArgBuffer(5, inInLinksStartIndexBuf)
	kernel.SetArgBuffer(6, inOutLinksStartIndexBuf)
	SetArgFloat64(kernel, 7, dampingFactor)
	SetArgFloat64(kernel, 8, tolerance)
	kernel.SetArgBuffer(9, outRankBuf)
	kernel.SetArgBuffer(10, outEntropyBuf)
	kernel.SetArgBuffer(11, outKarmaBuf)
	kernel.SetArgBuffer(12, outLuminocityBuf)

	fmt.Println("Memory allocation and transfer (to device): %s", time.Since(start).String())

	start = time.Now()

	_, err = queue.EnqueueNDRangeKernel(kernel, nil, []int{256}, []int{32}, nil)
	if err != nil {
		return nil, nil, nil, err
	}

	fmt.Println("Rank computation: %s", time.Since(start).String())

	start = time.Now()

	_, err = EnqueueReadBufferFloat64(queue, outRankBuf, true, 0, rank, nil)
	if err != nil {
		return nil, nil, nil, err
	}

	_, err = EnqueueReadBufferFloat64(queue, outEntropyBuf, true, 0, entropy, nil)
	if err != nil {
		return nil, nil, nil, err
	}

	_, err = EnqueueReadBufferFloat64(queue, outKarmaBuf, true, 0, karma, nil)
	if err != nil {
		return nil, nil, nil, err
	}

	_, err = EnqueueReadBufferFloat64(queue, outLuminocityBuf, true, 0, luminosity, nil)
	if err != nil {
		return nil, nil, nil, err
	}

	fmt.Println("Memory transfer (from device): %s", time.Since(start).String())

	return rank, entropy, karma, nil
}

func main() {
	device, err := getFirstOpenCLDevice()
	if err != nil {
		fmt.Println(err)
		return
	}

	fmt.Printf("OpenCL device name: %s", device.Name())

	dev := []*cl.Device{device}
	clContext, err := cl.CreateContext(dev)
	if err != nil {
		fmt.Println(err)
		return
	}

	queue, err := clContext.CreateCommandQueue(device, 0)
	if err != nil {
		fmt.Println(err)
		return
	}

	kernel, err := build_program_and_get_kernel(device, clContext)
	if err != nil {
		fmt.Println(err)
		return
	}

	rank, entropy, karma, err := calculate_rank(clContext, queue, kernel, nil, 0, nil, nil, nil, nil, nil, nil, 0, 0)
	if err != nil {
		fmt.Println(err)
		return
	}

	fmt.Println("Rank calculation result: ranks %d, entropy %d, karma %d", len(rank), len(entropy), len(karma))
}
