import json

from surprise import SVD
from surprise import Dataset
from surprise.model_selection import cross_validate
import pandas as pd
from scipy import spatial

from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/getNearestRelocation', methods=['post'])
def getNearestRelocation():
    locations = []
    content = request.get_json(silent=True)
    traffic = {
        'low': 0,
        'moderate': 1,
        'high': 2
    }
    rent = {
        'under $200': 0,
        '$200 to $700': 1,
        'above $700': 2
    }
    education = {
        'graduate': 0,
        'high school': 1,
        'college': 2
    }
    taxes = {
        'under 5%': 0,
        '5% to 15%': 1,
        'above 15%': 2
    }
    crime = {
        'low': 0,
        'moderate': 1,
        'high': 2
    }
    pop_density = {
        'high': 2,
        'moderate': 1,
        'low': 0
    }
    living_expense = {
        'under 40k': 0,
        '40k to 100k': 1,
        'more than 100k': 2
    }
    dist_from_cities = {
        'under 10 miles': 0,
        '10 to 50 miles': 1,
        'more than 100 miles': 2
    }
    df = pd.read_csv('Relocation Cities.csv')
    for index, row in df.iterrows():
        t = [
                row['Taxes'],
                row['crime rate'],
                row['Housing Costs (Rent)'],
                traffic[row['Traffic'].lower().strip()],
                education[row['Standard of Education'].lower().strip()],
                pop_density[row['Population density'].lower().strip()],
                living_expense[row['Living Expenses'].lower().strip()],
                dist_from_cities[row['distance from other cities'].lower().strip()]
            ]
        weather = row['weather'].split(' ')
        local_transport = row['access of local transport'].split()
        for i in range(0, len(weather)):
            weather[i] = weather[i].lower().strip()

        if 'cold' in weather:
            t.append(1)
        else:
            t.append(-1)

        if 'humid' in weather:
            t.append(1)
        else:
            t.append(-1)

        if 'dry' in weather:
            t.append(1)
        else:
            t.append(-1)

        if 'warm' in weather:
            t.append(1)
        else:
            t.append(-1)

        if 'hot' in weather:
            t.append(1)
        else:
            t.append(-1)

        for i in range(0, len(local_transport)):
            local_transport[i] = local_transport[i].lower().strip()

        if 'road' in local_transport:
            t.append(1)
        else:
            t.append(-1)

        if 'water' in local_transport:
            t.append(1)
        else:
            t.append(-1)

        if 'metro' in local_transport:
            t.append(1)
        else:
            t.append(-1)

        if 'air' in local_transport:
            t.append(1)
        else:
            t.append(-1)

        locations.append({
            'city': row['City'].lower().strip(),
            'taxes': row['Taxes'],
            'crime_rate': row['crime rate'],
            'image_url': row['Image link'],
            'rent': row['Housing Costs (Rent)'],
            'traffic': row['Traffic'].lower().strip(),
            'standard_of_education': row['Standard of Education'].lower().strip(),
            'population_density': row['Population density'].lower().strip(),
            'living_expense': row['Living Expenses'].lower().strip(),
            'dist_cities': row['distance from other cities'].lower().strip(),
            'vector': t
        })

    vector_cmp = [
        content['taxes'],
        content['crime_rate'],
        content['rent'],
        traffic[content['traffic'].lower().strip()],
        education[content['standard_of_education'].lower().strip()],
        pop_density[content['population_density'].lower().strip()],
        living_expense[content['living_expenses'].lower().strip()],
        dist_from_cities[content['distance_from_other_cities'].lower().strip()]
    ]
    weather = content['weather']
    local_transport = content['access_of_local_transport']
    for i in range(0, len(weather)):
        weather[i] = weather[i].lower().strip()

    if 'cold' in weather:
        vector_cmp.append(1)
    else:
        vector_cmp.append(-1)

    if 'humid' in weather:
        vector_cmp.append(1)
    else:
        vector_cmp.append(-1)

    if 'dry' in weather:
        vector_cmp.append(1)
    else:
        vector_cmp.append(-1)

    if 'warm' in weather:
        vector_cmp.append(1)
    else:
        vector_cmp.append(-1)

    if 'hot' in weather:
        vector_cmp.append(1)
    else:
        vector_cmp.append(-1)

    for i in range(0, len(local_transport)):
        local_transport[i] = local_transport[i].lower().strip()

    if 'road' in local_transport:
        vector_cmp.append(1)
    else:
        vector_cmp.append(-1)

    if 'water' in local_transport:
        vector_cmp.append(1)
    else:
        vector_cmp.append(-1)

    if 'metro' in local_transport:
        vector_cmp.append(1)
    else:
        vector_cmp.append(-1)

    if 'air' in local_transport:
        vector_cmp.append(1)
    else:
        vector_cmp.append(-1)

    for i in range(0, len(locations)):
        locations[i]['similarity'] = 1 - spatial.distance.cosine(locations[i]['vector'], vector_cmp)
    print(locations)
    res = []
    locations = sorted(locations, key=lambda i: i['similarity'], reverse=True)
    for i in range(0, 3):
        res.append(locations[i])

    for i in range(0, len(res)):
        del res[i]['vector']

    return jsonify(sorted(res, key = lambda i: i['similarity'], reverse=True)) , 200

@app.route('/getNearestVacation', methods=['post'])
def getNearestVacation():
    locations = []
    content = request.get_json(silent=True)
    budget = {
        'high': 2,
        'medium': 1,
        'low': 0
    }
    weather = {
        'cold': 0,
        'warm': 1,
        'humid': 2,
        'tropical': 3
    }
    historical = {
        'yes': 1,
        'no': 0
    }
    terrain = {
        'flat': 0,
        'mountain': 1,
        'river': 2,
        'coastal': 3,
        'island': 4,
        'forest': 5,
        'desert': 6
    }

    family_friendly = {
        'yes': 1,
        'no': 0
    }

    cuisine = {
        'continental': 0,
        'local': 1,
    }

    transport = {
        'public': 0,
        'rental': 1
    }

    social_env = {
        'friendly': 0,
        'acceptable': 1
    }

    party = {
        'yes': 1,
        'no': 0
    }
    season = {
        'peak season': 0,
        'off-season': 1
    }

    accomodation = {
        'hotel': 0,
        'homestay': 1,
        'hostel': 2
    }

    df = pd.read_csv('Vacation_citites.csv')
    for index, row in df.iterrows():
        locations.append({
            'location': row['Locations'].lower().strip(),
            'budget':row['Budget'].lower().strip(),
            'weather': row['Weather'].lower().strip(),
            'image_url': row['Image Url'].lower().strip(),
            'historical_places': row['Historical places'].lower().strip(),
            'type_of_terrain': row['Type of Terrain'].lower().strip(),
            'family_friendly': row['Family Friendly'].lower().strip(),
            'party_places': row['Party Places'].lower().strip(),
            'cuisine': row['Cuisine'].lower().strip(),
            'transport': row['Local Transport'].lower().strip(),
            'social_env': row['Social Enviroment'].lower().strip(),
            'season': row['Season'].lower().strip(),
            'accomodation': row['Accomodation'].lower().strip(),
            'vector':[ budget[row['Budget'].lower().strip()],
            weather[row['Weather'].lower().strip()],
            historical[row['Historical places'].lower().strip()],
            terrain[row['Type of Terrain'].lower().strip()],
             family_friendly[row['Family Friendly'].lower().strip()],
             party[row['Party Places'].lower().strip()],
            cuisine[row['Cuisine'].lower().strip()],
            transport[row['Local Transport'].lower().strip()],
           social_env[row['Social Enviroment'].lower().strip()],
            season[row['Season'].lower().strip()],
            accomodation[row['Accomodation'].lower().strip()]]
        })
    vector_cmp = [
        budget[content['budget'].lower().strip()],
        weather[content['weather'].lower().strip()],
        historical[content['historical'].lower().strip()],
        terrain[content['terrain'].lower().strip()],
        family_friendly[content['family_friendly'].lower().strip()],
        party[content['party'].lower().strip()],
        cuisine[content['cuisine'].lower().strip()],
        transport[content['transport'].lower().strip()],
        social_env[content['social_env'].lower().strip()],
        season[content['season'].lower().strip()],
        accomodation[content['accomodation'].lower().strip()]
    ]

    for i in range(0, len(locations)):
        locations[i]['similarity'] = 1 - spatial.distance.cosine(locations[i]['vector'], vector_cmp)

    locations = sorted(locations, key=lambda i: i['similarity'], reverse=True)
    res = []
    for i in range(0, 3):
        res.append(locations[i])

    for i in range(0, len(res)):
        del res[i]['vector']
    final_res = {
        'city1': res[0],
        'city2': res[1]
    }
    #print(locations)
    return jsonify(final_res) , 200

if __name__ == '__main__':
    # run() method of Flask class runs the application
    # on the local development server.
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
