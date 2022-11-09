# -*- coding: utf-8 -*-
"""
Title: PHYS20161 Final assignment: Z^0 boson


This code reads in data from two files and brings them together into the same
2D array. Then it filters the data, removing any errors, nans, negative values
and outliers.
The code then performs a minimised chi square fit on data described by the
cross section function. From here a plot is made of energy against cross section
with the previous fit and values for mass, width and lifetime are found.
Subsequently, the code creates a contour plot for mass and width values and
finds the uncertanties for mass, width and lifetime values.


Arnau Duran Mayol 13/12/2021
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin


FILE_NAME_1 = 'z_boson_data_1.csv'
FILE_NAME_2 = 'z_boson_data_2.csv'
PARTIAL_WIDTH = 83.91*10**-3  #GeV
HBAR = 6.582119569*10**-25 #GeV



def read_data(file_name_1, file_name_2):
    """
    Reads both data files and puts them together

    Args
    ----------
    file_name_1: string
    file_name_2: string

    Returns
    -------
    2D numpy array of floats
    """
    data_1 = np.genfromtxt(file_name_1, delimiter=',', comments='%')
    data_2 = np.genfromtxt(file_name_2, delimiter=',', comments='%')
    data = np.vstack((data_1, data_2))
    return data

def filter_data(data):
    """
    Returns the filtered data, without any nans or negative
    numbers. It sorts the data depending on energy aswell.

    Args
    ----------
    data : 2D numpy array of floats

    Returns
    -------
    2D numpy array of floats
    """
    data = data[np.where(~np.isnan(data).any(axis=1))]
    data = data[np.where(data[:, 1] > 0)]
    data = data[np.where(data[:, 2] > 0)]
    data = data[np.argsort(data[:, 0])]
    return data

def filter_outliers(data, predictions):
    '''
    Returns the filtered data without any outliers
    Args
    ----------
    data: 2D array of floats
    predictions: cross section function
    Returns
    -------
    data: 2D array of floats
    '''
    max_value = 0
    max_index = 0
    for index in range(len(data[:, 0])):
        if(np.abs(predictions[index] - data[index, 1])/data[index, 2]) > max_value: #
            max_value = np.abs(predictions[index] - data[index, 1])/data[index, 2]
            max_index = index
    data = np.vstack((data[0:max_index, :], data[max_index+1:, :]))
    return data

def cross_section_function(data, parameters):
    """
    Cross section function to be fitted
    cross section = 12π/mass^2*(E^2*partialwidth)/((E^2 − mass^2)^2 + mass^2width^2))

    Args
    ----
    paramaters: unknown mass and width
    PARTIAL_WIDTH: int

    Returns
    -------
    function depending on parameters and data
    """
    return (12*np.pi)/parameters[0]**2*(data[:, 0]**2
                                        *PARTIAL_WIDTH**2)/((data[:, 0]**2-
                                                             parameters[0]**2)**2+
                                                            parameters[0]**2*
                                                            parameters[1]**2)*0.3894*10**6

def reduced_chi_squared(parameters, data):
    '''
    Produces a reduced chi squared
    Args
    ----------
    degrees_of_freedom: integer of the length of the 2D array minus its dimensions
    predicions: uses previous function
    data: 2D array of floats
    parameters: unknown mass and width
    Returns
    -------
    reduced chi squared for the cross section function
    '''
    degrees_of_freedom = len(data) - 2
    predictions = cross_section_function(data, parameters)

    return np.sum(((data[:, 1] - predictions) / data[:, 2])**2)/degrees_of_freedom

def fit(data):
    '''
    Performs a fit of the data using fmin and the previous reduced chi squared
    Args
    ----------
    data : 2D array of floats

    Returns
    -------
    Appropiate fit for the data
    values for parameters

    '''

    return fmin(reduced_chi_squared, (90, 3), args=(data,), disp=0)

def plot(data, result, final_reduced_chi_squared):
    '''
    Creates a plot of energy against cross section with its uncertainties
    Args
    ----------
    data : 2D array of floats
    result : fitted parameters

    Returns
    -------
    Plot of data with its correct fit

    '''
    fig = plt.figure()
    ax_1 = fig.add_subplot(111)
    ax_1.errorbar(data[:, 0], data[:, 1], yerr=data[:, 2], fmt='.b')
    ax_1.plot(data[:, 0], cross_section_function(data, result), color='red')
    ax_1.set_title(r'Fit of energy against cross section for $Z^0$ Boson')
    ax_1.set_xlabel('Energy(GeV)')
    ax_1.set_ylabel(r'$\sigma$ (nb)')
    ax_1.legend([r'best fit with $\chi^2_R$=''{0:.3f}'.format(
        final_reduced_chi_squared)], loc='upper left')
    plt.savefig('Z_BOSON_PLOT.png', dpi=500)
    plt.show()

def contour_plot(width_values, mass_values, data):
    '''
    Performs an array mesh of width and mass values to then create a contour
    plot.  It also shows the region of chi2 +1.
    Args
    ---------
    mass_mesh:  2D array of floats
    width_mesh:  2D array of floats

    Returns
    -------
    Contour plot
    '''
    degrees_of_freedom = len(data) - 2
    mass_mesh = np.empty((0, len(mass_values)))
    for _ in range(len(width_values)):
        mass_mesh = np.vstack((mass_mesh, mass_values))
    width_mesh = np.empty((0, len(width_values)))
    for _ in range(len(mass_values)):
        width_mesh = np.vstack((width_mesh, width_values))
    width_mesh = np.transpose(width_mesh)
    chi_squared_mesh = np.zeros((100, 100))
    for k in range(len(mass_mesh)):
        for j in range(len(width_mesh)):
            chi_squared_mesh[k, j] = degrees_of_freedom*reduced_chi_squared((
                mass_mesh[k, j], width_mesh[k, j]), data)
    fig = plt.figure()
    ax_2 = fig.add_subplot(111)
    contourf = ax_2.contourf(mass_mesh, width_mesh, chi_squared_mesh, 10)
    contour_solution = ax_2.contour(mass_mesh, width_mesh, chi_squared_mesh >
                                    np.min(chi_squared_mesh+1), 1, colors='w')
    fig.colorbar(contourf)
    ax_2.set_title(r'${\chi}^2$ Contour of mass against width')
    ax_2.set_xlabel(r'Energy(Ge$V^2$)')
    ax_2.set_ylabel('Width(GeV)')
    contour_solution.collections[0].set_label(r'${\chi}^2_{min}$+1')
    ax_2.legend(facecolor='white')
    plt.savefig('contour_plot_mass_width.png', dpi=500)
    plt.show()

    return contour_solution

def uncertainties(contour_solution):
    '''
    Finds the uncertainties for mass and width
    Args
    ----------
    mass_values: array of floats
    width_values : array of floats
    data : 2D array of floats
    contour_solution : contour of the chi2+1 area

    Returns
    -------
    uncertainties_mass : float
    uncertainties_width : float
    '''

    for item in contour_solution.collections:
        for i in item.get_paths():
            vertex_values = i.vertices
            x_values = vertex_values[:, 0]
            y_values = vertex_values[:, 1]
    max_value_x = np.max(x_values)
    min_value_x = np.min(x_values)
    max_value_y = np.max(y_values)
    min_value_y = np.min(y_values)
    uncertainty_mass = (max_value_x-min_value_x)/2
    uncertainty_width = (max_value_y-min_value_y)/2

    return uncertainty_mass, uncertainty_width

def main():
    '''
    Main code for programme. Reads data, filters outliers,
    performs a minimised chi squared fit, a plot and a contour plot
    to find uncertainties
    Returns
    -------
    mass: int
    width: int
    lifeteime: int
    reduced chi square: int
    uncertainty_mass: int
    uncertainty_width: int
    uncertainty_lifetime: int

    Raises
    ------
    OSError: if file not found

    '''
    mass_values = np.linspace(91.15, 91.21, 100)
    width_values = np.linspace(2.48, 2.54, 100)
    try:
        data = read_data(FILE_NAME_1, FILE_NAME_2)
        data = filter_data(data)
        result = fit(data)
        data = filter_outliers(data, cross_section_function(data, result))
        result = fit(data)
        data = filter_outliers(data, cross_section_function(data, result))
        result = fit(data)
        data = filter_outliers(data, cross_section_function(data, result))
        result = fit(data)
        lifetime = HBAR/result[1]
        final_reduced_chi_squared = reduced_chi_squared(result, data)
        plot(data, result, final_reduced_chi_squared)
        countour_solution = contour_plot(width_values, mass_values, data)
        uncertainty_mass = uncertainties(countour_solution)[0]
        uncertainty_width = uncertainties(countour_solution)[1]
        uncertainty_lifetime = uncertainty_width/result[1]*lifetime
        print('Results for Z Boson: \n Mass: {0:.4g} ± {4:.2f} GeV/c^2 \n Width: {1:.4g}'
              ' ± {5:.3f} GeV \n Lifetime: {2:.3g} ± {6:.1g} s \n Reduced chi squared: {3:.3f}'
              .format(result[0], result[1], lifetime, final_reduced_chi_squared, uncertainty_mass,
                      uncertainty_width, uncertainty_lifetime))
        return 0
    except OSError:
        print('File not found')
        return None
main()
