package main

import (
	"errors"
	"fmt"

	"github.com/charliehorse55/go-opencl/cl"
)

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

func main() {
	device, err := getFirstOpenCLDevice()
	if err != nil {
		fmt.Println(err)
	}

	fmt.Printf("OpenCL device name: %s", device.Name())
}
