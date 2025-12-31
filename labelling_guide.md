# Vehicle Image Classification – Labeling Guide
## 1. Purpose

This labeling guide defines the official rules and criteria for assigning class labels to vehicle images used in the Vehicle Image Classification (13 Classes) project.
Its objective is to ensure consistent, reproducible, and high-quality labeling across the entire dataset.

## 2. Scope

Input: Still images

Content: Exactly one vehicle per image

Camera angles: Variable (front, rear, side, oblique)

Task: Assign one and only one class label to each image

## 3. General Labeling Rules

Single Vehicle Requirement
Each image must contain exactly one clearly identifiable vehicle.
Images containing multiple vehicles must be excluded.

Visibility and Quality
The vehicle must be sufficiently visible to determine its structural category.
Images with severe blur, extreme occlusion, or poor lighting that prevents identification must be rejected.

Structure Over Appearance
Label vehicles based on structural characteristics (vehicle type, axles, trailers), not perceived size or distance.

No Guessing Policy
If the correct class cannot be determined with high confidence, the image must be excluded.
Ambiguous images must never be labeled by assumption.

Background Irrelevance
Road type, lane count, surrounding vehicles, or camera position must not influence labeling decisions.

## 4. Class Definitions
### Class 01 – Motorcycle

Two-wheeled motorized vehicle

May include rider or passenger

Includes: motorcycles, scooters
Excludes: bicycles, tricycles

### Class 02 – Passenger Car

Four-wheeled passenger vehicle

No trailer attached

Includes: sedans, hatchbacks, coupes, taxis
Excludes: SUVs or cars towing trailers

### Class 03 – Four tire, single unit

Four-tire, single-unit (rigid) vehicle

No trailer attached

Includes: pickups, small rigid vans and utility/ambulance vehicles
Excludes: multi-axle trucks and articulated tractor-trailer combinations

### Class 04 – Bus

Large passenger vehicle (single-unit bus)

Multiple side windows and higher roofline typical of buses

No trailer attached

Includes: city buses, intercity coaches, single-deck buses
Excludes: minibuses and articulated/multi-unit buses

### Class 05 – Two axle, six tire, single unit

Two-axle, six-tire single-unit vehicles

Rigid single-unit vehicles with two axles and six tires

Includes: medium-duty rigid trucks and larger single-unit service vehicles
Excludes: articulated tractor-trailer combinations

### Class 06 – Three axle, single unit

Three-axle single-unit vehicles

Rigid trucks with three axles and no trailer attached

Includes: cement mixers, larger single-unit box trucks and dump trucks
Excludes: tractor-trailer combinations

### Class 07 – Four or more axle, single unit

Four or more axle, single-unit vehicles

Rigid trucks with four or more axles and no trailers attached

Includes: heavy dump trucks and large rigid haulage units
Excludes: articulated tractor-trailer combinations

### Class 08 – Four or less axle, single trailer

Four or less axle, single-trailer configurations

Tractor or rigid vehicle coupled with a single trailer where total axles are four or less

Includes: light single-trailer trucks and short-semitrailer combinations
Excludes: multi-trailer road-train configurations

### Class 09 – 5-Axle tractor semitrailer

Five-axle tractor semitrailer configurations

Standard tractor unit coupled with a single semitrailer, commonly totaling five axles

Includes: typical articulated 5-axle semitrailers
Excludes: double or multi-trailer configurations

### Class 10 – Six or more axle, single trailer

Six or more axle, single-trailer configurations

Tractor + single trailer combinations where the total axle count is six or more

Includes: heavy articulated trucks with high axle counts
Excludes: multi-trailer road-train configurations

### Class 11 – Five or less axle, multi trailer

Five or less axle, multi-trailer configurations

Multi-trailer combinations where the total axle count is five or less

Includes: light multi-trailer setups (two connected trailers) with combined axles ≤ 5
Excludes: single-trailer and rigid single-unit vehicles

### Class 12 – Six axle, multi-trailer

Six-axle, multi-trailer configurations

Multi-trailer combinations where the combined axle count is six

Includes: multi-trailer combinations with total axles = 6
Excludes: single-unit rigid trucks

### Class 13 – Seven or more axle, multi-trailer

Seven or more axle, multi-trailer road-train configurations

Large multi-trailer combinations with seven or more axles

Includes: long road-trains and heavy haulage multi-trailer setups
Excludes: standard single-trailer or rigid trucks

## 5. Ambiguous and Rejection Criteria

An image must be excluded if:

The trailer or axle configuration is partially outside the frame and cannot be confirmed

Vehicle structure cannot be determined within reasonable confidence

The vehicle occupies too little of the image to identify reliably